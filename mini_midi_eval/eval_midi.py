from midi_eval.eval_hf_helpers import run_full_evaluation_hf
from midi_eval.eval_mini_helpers import run_full_evaluation_mini


def run_full_evaluation(
    midi_root: str,
    meta_path: str = "midicaps_meta/test_split.json",
    max_examples: int = 256,
    batch_size: int = 4,
    max_target_len: int = 1024,
    max_gen_len: int = 200,
    model_path: str | None = None,
    tokenizer_path: str | None = None,
    model_type: str = "hf",
):
    if model_type == "hf":
        return run_full_evaluation_hf(
            midi_root=midi_root,
            meta_path=meta_path,
            max_examples=max_examples,
            batch_size=batch_size,
            max_target_len=max_target_len,
            max_gen_len=max_gen_len,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
        )
    elif model_type == "mini":
        return run_full_evaluation_mini(
            midi_root=midi_root,
            meta_path=meta_path,
            max_examples=max_examples,
            batch_size=batch_size,
            max_target_len=max_target_len,
            max_gen_len=max_gen_len,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Text2Midi (HF) or MiniText2MIDI on MidiCaps test split with "
            "NLL/PPL, text–MIDI retrieval, BLEU-4, and feature-wise metrics."
        )
    )
    parser.add_argument(
        "--midi_root",
        type=str,
        required=True,
        help="Root folder containing Lakh MIDI (e.g. midicaps_lmd, which has lmd_full/...).",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default="midicaps_meta/test_split.json",
        help="Path to local MidiCaps test_split.json.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=256,
        help="Maximum number of test examples to evaluate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for teacher-forced and embedding computations.",
    )
    parser.add_argument(
        "--max_target_len",
        type=int,
        default=1024,
        help="Max MIDI token length (truncate longer sequences).",
    )
    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=200,
        help="Maximum length of generated MIDI token sequences.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "Path to model weights. "
            "For hf: optional path to .bin file (if None, uses HF baseline); "
            "for mini: REQUIRED path to .pt MiniText2MIDI weights."
        ),
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional path to vocab_remi.pkl (if None, uses HF).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["hf", "mini"],
        default="hf",
        help="Which model to evaluate: 'hf' for original Text2MIDI, 'mini' for MiniText2MIDI.",
    )

    args = parser.parse_args()

    run_full_evaluation(
        midi_root=args.midi_root,
        meta_path=args.meta_path,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        max_target_len=args.max_target_len,
        max_gen_len=args.max_gen_len,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        model_type=args.model_type,
    )
