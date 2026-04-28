from __future__ import annotations

import argparse
from pathlib import Path

from chunking import process_course, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build RAG chunks from per-course JSON under retrieval_corpus/<course_id>/."
    )
    p.add_argument(
        "retrieval_root",
        type=Path,
        nargs="?",
        default=Path("retrieval_corpus"),
        help="Parent folder containing course subdirs (default: retrieval_corpus)",
    )
    p.add_argument(
        "--course",
        type=str,
        default="cogsci_c127",
        help="Course id (subfolder name). Chunks include metadata['course_id'] for scoped retrieval.",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("retrieval_corpus/cogsci_c127/chunks/lecture_chunks.jsonl"),
        help="Output JSONL path",
    )
    p.add_argument(
        "--strategy",
        choices=["speech_anchored", "time_window"],
        default="time_window",
        help="speech_anchored: one chunk per ASR segment; time_window: fixed windows",
    )
    p.add_argument(
        "--window-s",
        type=float,
        default=45.0,
        help="Window length in seconds (time_window only)",
    )
    p.add_argument(
        "--ocr-min-len",
        type=int,
        default=12,
        help="Min OCR line length to include (reduces menu chrome noise)",
    )
    args = p.parse_args()
    items = process_course(
        args.retrieval_root.resolve(),
        args.course,
        strategy=args.strategy,
        window_s=args.window_s,
        ocr_min_len=args.ocr_min_len,
    )
    out = args.out.resolve()
    write_jsonl(out, items)
    print(f"Wrote {len(items)} chunks to {out}")


if __name__ == "__main__":
    main()
