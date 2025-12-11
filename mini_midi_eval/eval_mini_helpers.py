# eval_mini.py

import pickle

import torch
from transformers import T5Tokenizer
from huggingface_hub import hf_hub_download

from mini_midi.minitext2midi import MiniText2MIDI

from midi_eval.eval_helpers import (
    get_device,
    load_midicaps_test_split_local,
    tokenize_midi_files,
    pad_token_batch,
    compute_text_embeddings,
    compute_retrieval_metrics,
    compute_nll_and_ppl,
    evaluate_generation_bleu_and_features,
)


def load_minitext2midi(
    model_path: str,
    remi_vocab_size: int,
    max_remi_tokens: int,
    device: torch.device,
) -> MiniText2MIDI:
    """Load MiniText2MIDI model."""
    model = MiniText2MIDI(
        remi_vocab_size=remi_vocab_size,
        max_remi_tokens=max_remi_tokens,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _mini_forward(
    model: MiniText2MIDI,
    src_ids: torch.Tensor,
    src_mask: torch.Tensor,
    tgt_in: torch.Tensor,
) -> torch.Tensor:
    """Forward wrapper for NLL/PPL."""
    out = model(
        text_input_ids=src_ids,
        attention_mask=src_mask,
        decoder_input_ids=tgt_in,
        labels=None,
    )
    return out["logits"]


def generate_mini(
    model: MiniText2MIDI,
    src_ids: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int = 200,
    temperature: float = 1.0,
    bos_token_id: int = 1,
) -> torch.Tensor:
    """Autoregressive MiniText2MIDI sampling."""
    model.eval()

    if src_ids.dim() != 2:
        raise RuntimeError("src_ids should be 2D (B, L)")

    B = src_ids.size(0)
    device = src_ids.device

    tgt_fin = torch.full(
        (B, 1),
        bos_token_id,
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_len):
        out = model(
            text_input_ids=src_ids,
            attention_mask=src_mask,
            decoder_input_ids=tgt_fin,
            labels=None,
        )
        logits = out["logits"][:, -1, :]
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        tgt_fin = torch.cat([tgt_fin, next_tokens], dim=1)

    return tgt_fin[:, 1:]


def _mini_generate(
    model: MiniText2MIDI,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    max_gen_len: int,
    bos_id: int,
) -> list[int]:
    """Generation wrapper for BLEU/features."""
    gen = generate_mini(
        model,
        input_ids,
        attn_mask,
        max_len=max_gen_len,
        temperature=1.0,
        bos_token_id=bos_id,
    )
    return gen[0].tolist()


def compute_midi_embeddings_mini(
    model: MiniText2MIDI,
    text_tokenizer: T5Tokenizer,
    captions: list[str],
    midi_tokens: list[torch.Tensor],
    device: torch.device,
    pad_id: int,
    batch_size: int = 4,
    max_src_len: int = 512,
) -> torch.Tensor:
    """Mean-pooled MiniText2MIDI decoder states."""
    midi_embs: list[torch.Tensor] = []
    model.eval()

    for start in range(0, len(captions), batch_size):
        end = min(start + batch_size, len(captions))
        batch_caps = captions[start:end]
        batch_toks = midi_tokens[start:end]

        non_empty = [i for i, t in enumerate(batch_toks) if t.numel() > 0]
        if not non_empty:
            continue

        batch_caps = [batch_caps[i] for i in non_empty]
        batch_toks = [batch_toks[i] for i in non_empty]

        inputs = text_tokenizer(
            batch_caps,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_src_len,
        )
        src_ids = inputs["input_ids"].to(device)
        src_mask = inputs["attention_mask"].to(device)

        tgt_full, tgt_mask_full = pad_token_batch(batch_toks, pad_id=pad_id)
        tgt_full = tgt_full.to(device)
        tgt_mask_full = tgt_mask_full.to(device)

        with torch.no_grad():
            enc_out = model.encoder(
                input_ids=src_ids,
                attention_mask=src_mask,
            ).last_hidden_state
            memory = model.enc_proj(enc_out)

            B, T = tgt_full.shape
            x = model.decoder.embedding(tgt_full) + model.decoder.pos_emb[:, :T, :]

            causal_mask = torch.triu(
                torch.ones(T, T, device=device), diagonal=1
            ).bool()

            tgt_padding_mask = ~tgt_mask_full

            for layer in model.decoder.layers:
                x = layer(
                    x,
                    memory,
                    tgt_mask=causal_mask,
                    tgt_padding_mask=tgt_padding_mask,
                )

            dec_out = x

            mask = tgt_mask_full.unsqueeze(-1)
            summed = (dec_out * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = summed / lengths

        midi_embs.append(pooled.cpu())

    return torch.cat(midi_embs, dim=0)


def run_full_evaluation_mini(
    midi_root: str,
    meta_path: str = "midicaps_meta/test_split.json",
    max_examples: int = 256,
    batch_size: int = 4,
    max_target_len: int = 1024,
    max_gen_len: int = 200,
    model_path: str | None = None,
    tokenizer_path: str | None = None,
):
    """Run full MiniText2MIDI evaluation."""
    device = get_device()
    print(f"Using device: {device}")

    if tokenizer_path is None:
        tokenizer_path = hf_hub_download(
            repo_id="amaai-lab/text2midi",
            filename="vocab_remi.pkl",
        )

    with open(tokenizer_path, "rb") as f:
        r_tok = pickle.load(f)

    pad_id = getattr(r_tok, "pad_token_id", 0)
    bos_id = getattr(r_tok, "bos_token_id", 1)
    eos_id = getattr(r_tok, "eos_token_id", None)

    if model_path is None:
        raise ValueError("model_path must be provided for MiniText2MIDI (.pt file).")

    model = load_minitext2midi(
        model_path=model_path,
        remi_vocab_size=len(r_tok),
        max_remi_tokens=max_target_len,
        device=device,
    )

    text_tok = T5Tokenizer.from_pretrained("google/flan-t5-base")

    print("Vocab size:", len(r_tok))

    split = load_midicaps_test_split_local(
        meta_path=meta_path,
        max_examples=max_examples,
    )
    captions = split["captions"]
    locations = split["locations"]
    print(f"Loaded {len(captions)} MidiCaps test examples from {meta_path}.")

    midi_tokens = tokenize_midi_files(
        midi_root=midi_root,
        locations=locations,
        r_tokenizer=r_tok,
        max_target_len=max_target_len,
    )

    filtered_caps, filtered_tokens, filtered_locs = [], [], []
    for cap, tok, loc in zip(captions, midi_tokens, locations):
        if tok.numel() == 0:
            continue
        filtered_caps.append(cap)
        filtered_tokens.append(tok)
        filtered_locs.append(loc)

    captions = filtered_caps
    midi_tokens = filtered_tokens
    locations = filtered_locs
    print(f"After filtering, using {len(captions)} examples for evaluation.")

    if not captions:
        print("[WARN] No valid MIDI examples after filtering; aborting evaluation.")
        return

    avg_nll, ppl = compute_nll_and_ppl(
        forward_fn=_mini_forward,
        model=model,
        text_tokenizer=text_tok,
        captions=captions,
        midi_tokens=midi_tokens,
        device=device,
        pad_id=pad_id,
        batch_size=batch_size,
    )
    print(f"[NLL]    Mean per-sequence NLL: {avg_nll:.4f}")
    print(f"[PPL]    Perplexity:            {ppl:.4f}")

    text_embs = compute_text_embeddings(
        model=model,
        text_tokenizer=text_tok,
        captions=captions,
        device=device,
        batch_size=batch_size,
    )
    midi_embs = compute_midi_embeddings_mini(
        model=model,
        text_tokenizer=text_tok,
        captions=captions,
        midi_tokens=midi_tokens,
        device=device,
        pad_id=pad_id,
        batch_size=batch_size,
    )
    recalls, med_rank = compute_retrieval_metrics(text_embs, midi_embs)
    print("[Retrieval] Text–MIDI retrieval:")
    for k, v in recalls.items():
        print(f"  {k}: {v*100:.2f}%")
    print(f"  Median Rank: {med_rank:.2f}")

    meta = [{"caption": cap, "location": loc} for cap, loc in zip(captions, locations)]

    bleu4, feature_stats = evaluate_generation_bleu_and_features(
        generate_fn=_mini_generate,
        model=model,
        text_tokenizer=text_tok,
        r_tokenizer=r_tok,
        meta=meta,
        midi_root=midi_root,
        device=device,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        max_gen_len=max_gen_len,
        max_examples=max_examples,
        midi_tokens_ref=midi_tokens,
    )

    print(f"[BLEU-4] Mean BLEU-4 (summary): {bleu4:.4f}")
    print(
        "[Features] TB={TB:.2f}%, TBT={TBT:.2f}%, CK={CK:.2f}%, CKD={CKD:.2f}% "
        "(N={count})".format(**feature_stats)
    )
