import os
import math
import json
from pathlib import Path
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import T5Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from music21 import converter

TEMPO_BINS = [40, 60, 70, 90, 110, 140, 160, 210]


def get_device() -> torch.device:
    """Select CUDA, MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_midicaps_test_split_local(meta_path: str, max_examples: int | None = None):
    """Load captions + MIDI locations."""
    with open(meta_path, "r") as f:
        data = json.load(f)

    if max_examples is not None:
        data = data[:max_examples]

    return {
        "captions": [x["caption"] for x in data],
        "locations": [x["location"] for x in data],
    }


def tokenize_midi_files(midi_root: str, locations: list[str], r_tokenizer, max_target_len: int = 1024):
    """Tokenize MIDI files into ID sequences."""
    seqs = []
    root = Path(midi_root)

    for loc in locations:
        midi_path = root / loc
        try:
            tokens = r_tokenizer(midi_path)
            ids = torch.tensor(tokens.ids, dtype=torch.long)
            if len(ids) > max_target_len:
                ids = ids[:max_target_len]
            seqs.append(ids)
        except Exception as e:
            print(f"[WARN] Failed {midi_path}: {e}")
            seqs.append(torch.empty(0, dtype=torch.long))

    return seqs


def pad_token_batch(seqs: list[torch.Tensor], pad_id: int):
    """Pad 1D token sequences."""
    if len(seqs) == 0:
        return torch.empty(0, 0, dtype=torch.long), torch.empty(0, 0, dtype=torch.bool)

    fixed = [torch.tensor([pad_id], dtype=torch.long) if s.numel() == 0 else s for s in seqs]
    padded = pad_sequence(fixed, batch_first=True, padding_value=pad_id)
    mask = padded.ne(pad_id)
    return padded, mask


def compute_nll_and_ppl(
    forward_fn: Callable[[object, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    model: object,
    text_tokenizer: T5Tokenizer,
    captions: list[str],
    midi_tokens: list[torch.Tensor],
    device: torch.device,
    pad_id: int,
    batch_size: int = 4,
    max_src_len: int = 512,
):
    """Compute teacher-forced NLL and PPL."""
    model.eval()
    n = len(captions)
    sum_nll = 0.0
    count = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        caps = captions[start:end]
        toks = midi_tokens[start:end]

        keep = [i for i, t in enumerate(toks) if t.numel() > 1]
        if not keep:
            continue

        caps = [caps[i] for i in keep]
        toks = [toks[i] for i in keep]

        inputs = text_tokenizer(
            caps, return_tensors="pt", padding=True, truncation=True, max_length=max_src_len
        )
        src_ids = inputs["input_ids"].to(device)
        src_mask = inputs["attention_mask"].to(device)

        tgt_full, _ = pad_token_batch(toks, pad_id)
        if tgt_full.size(1) < 2:
            continue

        tgt_in = tgt_full[:, :-1].to(device)
        tgt_out = tgt_full[:, 1:].to(device)
        mask_out = tgt_out.ne(pad_id)

        with torch.no_grad():
            logits = forward_fn(model, src_ids, src_mask, tgt_in)

        B, Tm1, V = logits.size()
        losses = F.cross_entropy(
            logits.reshape(B * Tm1, V),
            tgt_out.reshape(B * Tm1),
            reduction="none",
            ignore_index=pad_id,
        ).view(B, Tm1)

        counts = mask_out.sum(dim=1)
        good = counts > 0
        if good.sum() == 0:
            continue

        seq_nll = (losses[good].sum(dim=1) / counts[good]).tolist()
        sum_nll += sum(seq_nll)
        count += len(seq_nll)

    if count == 0:
        return float("nan"), float("nan")

    avg = sum_nll / count
    return avg, math.exp(avg)


def compute_text_embeddings(
    model: object,
    text_tokenizer: T5Tokenizer,
    captions: list[str],
    device: torch.device,
    batch_size: int = 8,
    max_src_len: int = 512,
):
    """Mean-pooled encoder embeddings."""
    embs = []
    model.eval()

    for start in range(0, len(captions), batch_size):
        end = min(start + batch_size, len(captions))
        batch = captions[start:end]

        inputs = text_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=max_src_len
        )
        src_ids = inputs["input_ids"].to(device)
        src_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            enc = model.encoder(input_ids=src_ids, attention_mask=src_mask).last_hidden_state
            if hasattr(model, "enc_proj"):
                enc = model.enc_proj(enc)

            mask = src_mask.unsqueeze(-1)
            summed = (enc * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = summed / lengths

        embs.append(pooled.cpu())

    return torch.cat(embs, dim=0)


def compute_retrieval_metrics(text_embs: torch.Tensor, midi_embs: torch.Tensor, ks=(1, 5, 10)):
    """Compute recall@k and median rank."""
    assert text_embs.shape == midi_embs.shape
    text = F.normalize(text_embs, dim=-1)
    midi = F.normalize(midi_embs, dim=-1)

    sims = midi @ text.t()
    N = sims.size(0)
    _, idxs = sims.sort(dim=-1, descending=True)

    ranks = []
    for i in range(N):
        pos = (idxs[i] == i).nonzero(as_tuple=False)
        ranks.append(int(pos[0, 0]) + 1 if pos.numel() else N)

    ranks_t = torch.tensor(ranks, dtype=torch.float32)
    recalls = {f"R@{k}": (ranks_t <= k).float().mean().item() for k in ks}
    return recalls, ranks_t.median().item()


def average_bleu4(references: list[torch.Tensor], hypotheses: list[list[int]], pad_id: int, bos_id: int, eos_id: int | None):
    """Mean BLEU-4 score."""
    smoothie = SmoothingFunction().method1
    scores = []

    def strip(seq):
        out = []
        for tok in seq:
            if tok == pad_id or tok == bos_id:
                continue
            if eos_id is not None and tok == eos_id:
                break
            out.append(tok)
        return out

    for ref_t, hyp in zip(references, hypotheses):
        ref = strip(ref_t.tolist())
        hyp = strip(hyp)
        if not ref or not hyp:
            scores.append(0.0)
            continue
        scores.append(sentence_bleu([ref], hyp, smoothing_function=smoothie))

    return sum(scores) / len(scores) if scores else 0.0


def tempo_bin_index(bpm: float) -> int:
    """Convert BPM to tempo bin."""
    for i, b in enumerate(TEMPO_BINS):
        if bpm < b:
            return i
    return len(TEMPO_BINS)


def extract_tempo_and_key_pc(midi_path: str):
    """Extract tempo + key from MIDI."""
    try:
        score = converter.parse(midi_path)
    except Exception:
        return None, None, None

    bpm = None
    try:
        mm = score.metronomeMarkBoundaries()
        if mm and mm[0][2] and hasattr(mm[0][2], "number"):
            bpm = float(mm[0][2].number)
    except Exception:
        pass

    key_pc, mode = None, None
    try:
        k = score.analyze("key")
        key_pc = k.tonic.pitchClass
        mode = k.mode
    except Exception:
        pass

    return bpm, key_pc, mode


def keys_equivalent(ref_pc, ref_mode, gen_pc, gen_mode):
    """Exact or relative major/minor match."""
    if None in (ref_pc, ref_mode, gen_pc, gen_mode):
        return False
    if ref_pc == gen_pc and ref_mode == gen_mode:
        return True
    if ref_mode == "major" and gen_mode == "minor" and gen_pc == (ref_pc + 9) % 12:
        return True
    if ref_mode == "minor" and gen_mode == "major" and gen_pc == (ref_pc + 3) % 12:
        return True
    return False


def evaluate_generation_bleu_and_features(
    generate_fn: Callable[[object, torch.Tensor, torch.Tensor, int, int], list[int]],
    model: object,
    text_tokenizer: T5Tokenizer,
    r_tokenizer,
    meta: list[dict],
    midi_root: str,
    device: torch.device,
    pad_id: int,
    bos_id: int,
    eos_id: int | None,
    max_gen_len: int = 200,
    max_examples: int | None = None,
    midi_tokens_ref: list[torch.Tensor] | None = None,
):
    """Compute BLEU-4 + tempo/key metrics."""
    import tempfile

    if midi_tokens_ref is None:
        raise ValueError("Missing reference tokens")

    ref_tensors = []
    hyps = []

    tb = tbt = ck = ckd = 0
    count = 0

    tmp = tempfile.mkdtemp(prefix="eval_gen_")

    for i, (ex, ref_tok) in enumerate(zip(meta, midi_tokens_ref)):
        if max_examples is not None and i >= max_examples:
            break

        caption = ex["caption"]
        midi_path = os.path.join(midi_root, ex["location"])

        if not os.path.exists(midi_path):
            continue
        if ref_tok.numel() == 0:
            continue

        ref = ref_tok.tolist()
        if bos_id is not None:
            ref = [bos_id] + ref
        if eos_id is not None:
            ref = ref + [eos_id]

        ref_tensors.append(torch.tensor(ref, dtype=torch.long))

        inputs = text_tokenizer(caption, return_tensors="pt", padding=True, truncation=True)
        inp = inputs.input_ids.to(device)
        mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            gen = generate_fn(model, inp, mask, max_gen_len, bos_id)

        if not gen:
            continue

        hyps.append(gen)

        rbpm, rpc, rmode = extract_tempo_and_key_pc(midi_path)

        try:
            midi_obj = r_tokenizer.decode(gen)
            gen_path = os.path.join(tmp, f"gen_{i}.mid")
            midi_obj.dump_midi(gen_path)
            gbpm, gpc, gmode = extract_tempo_and_key_pc(gen_path)
        except Exception:
            continue

        if None not in (rbpm, rpc, rmode, gbpm, gpc, gmode):
            count += 1
            if tempo_bin_index(rbpm) == tempo_bin_index(gbpm):
                tb += 1
            if abs(tempo_bin_index(rbpm) - tempo_bin_index(gbpm)) <= 1:
                tbt += 1
            if gpc == rpc and gmode == rmode:
                ck += 1
            if keys_equivalent(rpc, rmode, gpc, gmode):
                ckd += 1

    bleu = average_bleu4(ref_tensors, hyps, pad_id, bos_id, eos_id)
    print(f"[BLEU-4] {bleu:.4f}")

    if count == 0:
        print("[Features] No usable examples")
        return bleu, {"TB": 0, "TBT": 0, "CK": 0, "CKD": 0, "count": 0}

    results = {
        "TB": 100 * tb / count,
        "TBT": 100 * tbt / count,
        "CK": 100 * ck / count,
        "CKD": 100 * ckd / count,
        "count": count,
    }

    print(f"[Features] TB={results['TB']:.2f}%, TBT={results['TBT']:.2f}%, CK={results['CK']:.2f}%, CKD={results['CKD']:.2f}% (N={count})")

    return bleu, results
