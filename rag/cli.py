from __future__ import annotations

import argparse
from pathlib import Path

from rag.chunking import process_corpus, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build RAG chunks from frame_descriptions, ocr, and speech JSON."
    )
    p.add_argument(
        "corpus",
        type=Path,
        nargs="?",
        default=Path("retrieval_corpus"),
        help="Root folder with frame_descriptions/, ocr/, speech/ (default: retrieval_corpus)",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("retrieval_corpus/chunks/lecture_chunks.jsonl"),
        help="Output JSONL path",
    )
    p.add_argument(
        "--strategy",
        choices=["speech_anchored", "time_window"],
        default="speech_anchored",
        help="speech_anchored: one chunk per ASR segment; time_window: fixed windows",
    )
    p.add_argument(
        "--window-s",
        type=float,
        default=60.0,
        help="Window length in seconds (time_window only)",
    )
    p.add_argument(
        "--ocr-min-len",
        type=int,
        default=12,
        help="Min OCR line length to include (reduces menu chrome noise)",
    )
    args = p.parse_args()
    items = process_corpus(
        args.corpus.resolve(),
        strategy=args.strategy,
        window_s=args.window_s,
        ocr_min_len=args.ocr_min_len,
    )
    out = args.out.resolve()
    write_jsonl(out, items)
    print(f"Wrote {len(items)} chunks to {out}")


if __name__ == "__main__":
    main()
