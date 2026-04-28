#!/usr/bin/env python3
"""
Run lecture_questions.json through dense retrieval (local embeddings) + Hugging Face
chat completion, and report F1, exact match, recall, retrieval time, generation time,
and temporal retrieval hit (gold timestamp window overlaps any top-k chunk).

Dense retrieval needs ``numpy`` (project dependency). Local embeddings optionally use
``sentence-transformers`` (``uv sync --group rag``). If that is not installed, the
script defaults to Hugging Face Inference API embeddings (same token as chat).

HF chat + API embeddings: HUGGINGFACE_API_KEY or HF_TOKEN.

Edit prompts/lecture_qa_system.txt to tune the system prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluate import exact_match_score, f1_score, recall  # noqa: E402

ROUTER_CHAT = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_QUESTIONS = _ROOT / "retrieval_corpus/cogsci_c127/lecture_questions.json"
DEFAULT_PROMPT = _ROOT / "prompts/lecture_qa_system.txt"
DEFAULT_CHUNKS = _ROOT / "retrieval_corpus/chunks/lecture_chunks.jsonl"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def load_dotenv_simple(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def hf_token() -> str:
    return (os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN") or "").strip()


def call_hf_chat(
    *,
    model: str,
    system_prompt: str,
    user_content: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        ROUTER_CHAT,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {err_body}") from e
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices in response: {json.dumps(payload)[:2000]}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        raise RuntimeError(f"Missing message content: {json.dumps(payload)[:2000]}")
    return str(content).strip()


def build_user_message_closed_book(item: dict) -> str:
    lecture = item.get("lecture_name") or f"Lecture {item.get('lecture #', '')}"
    q = item.get("question", "").strip()
    return f"Lecture: {lecture}\n\nQuestion: {q}"


def build_user_message_rag(context: str, item: dict) -> str:
    lecture = item.get("lecture_name") or f"Lecture {item.get('lecture #', '')}"
    q = item.get("question", "").strip()
    return (
        f"Lecture: {lecture}\n\n"
        f"Context (retrieved passages):\n{context}\n\n"
        f"Question: {q}"
    )


def load_chunks(
    *,
    chunks_path: Path,
    build_chunks: bool,
    retrieval_root: Path,
    course_id: str,
    strategy: str,
    window_s: float,
    ocr_min_len: int,
) -> list[dict]:
    from rag.chunking import process_course

    if build_chunks:
        return process_course(
            retrieval_root.resolve(),
            course_id,
            strategy=strategy,  # type: ignore[arg-type]
            window_s=window_s,
            ocr_min_len=ocr_min_len,
        )
    if not chunks_path.is_file():
        raise FileNotFoundError(
            f"Chunk JSONL not found: {chunks_path}\n"
            "Run:  uv run python -m rag --course YOUR_COURSE -o retrieval_corpus/chunks/lecture_chunks.jsonl\n"
            "Or pass --build-chunks to build chunks in memory from retrieval_corpus (slower startup)."
        )
    from rag.retrieval import load_chunks_jsonl

    return load_chunks_jsonl(chunks_path)


def main() -> None:
    load_dotenv_simple(_ROOT / ".env")

    p = argparse.ArgumentParser(description="Eval lecture JSON QA via HF chat (+ optional dense retrieval).")
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="lecture_questions.json")
    p.add_argument("--system-prompt-file", type=Path, default=DEFAULT_PROMPT)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=0, help="Max questions (0 = all)")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--out", type=Path, default=None, help="JSONL log path")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Closed-book: send only lecture name + question (no local embedder).",
    )
    p.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS,
        help="Chunk JSONL (same format as rag chunking output)",
    )
    p.add_argument(
        "--build-chunks",
        action="store_true",
        help="Build chunks in memory via rag.chunking.process_course instead of reading --chunks",
    )
    p.add_argument("--retrieval-root", type=Path, default=_ROOT / "retrieval_corpus")
    p.add_argument("--course", type=str, default="", help="course_id filter (default: from first question)")
    p.add_argument(
        "--chunk-strategy",
        choices=["speech_anchored", "time_window"],
        default="speech_anchored",
        help="Only used when course spec uses triple pipeline (--build-chunks)",
    )
    p.add_argument("--window-s", type=float, default=60.0, help="time_window strategy only")
    p.add_argument("--ocr-min-len", type=int, default=12)
    p.add_argument(
        "--embedder-backend",
        choices=["auto", "local", "huggingface"],
        default="auto",
        help="auto: sentence-transformers if installed, else HF Inference API embeddings",
    )
    p.add_argument(
        "--embedder-model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Local sentence-transformers model (--embedder-backend local or auto)",
    )
    p.add_argument(
        "--api-embedder-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF Inference API model id (--embedder-backend huggingface or auto fallback)",
    )
    p.add_argument("--k", type=int, default=8, help="Top-k chunks per question")
    p.add_argument("--max-context-chars", type=int, default=12000)
    args = p.parse_args()

    use_retrieval = not args.no_retrieval
    token = hf_token()
    if not args.dry_run and not token:
        print("Set HUGGINGFACE_API_KEY or HF_TOKEN in the environment or .env", file=sys.stderr)
        sys.exit(1)

    system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
    if not system_prompt:
        print("System prompt file is empty.", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(args.questions.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        print("questions file must be a JSON array", file=sys.stderr)
        sys.exit(1)
    items: list[dict] = raw
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    course_id = args.course.strip() or str(items[0].get("course_id") or "cogsci_c127")

    print(f"Questions: {args.questions} ({len(items)} items)")
    print(f"Model: {args.model}")
    print(f"System prompt: {args.system_prompt_file}")
    print(f"Retrieval: {'on' if use_retrieval else 'off (closed-book)'}")
    if use_retrieval:
        print(f"  chunks: {args.chunks}  build_in_memory={args.build_chunks}")
        print(
            f"  course_id={course_id}  embedder_backend={args.embedder_backend}  "
            f"local_model={args.embedder_model}  api_model={args.api_embedder_model}  k={args.k}"
        )

    retriever = None
    lecture_ids_in_corpus: set[str] = set()
    if use_retrieval and not args.dry_run:
        try:
            chunks = load_chunks(
                chunks_path=args.chunks,
                build_chunks=args.build_chunks,
                retrieval_root=args.retrieval_root,
                course_id=course_id,
                strategy=args.chunk_strategy,
                window_s=args.window_s,
                ocr_min_len=args.ocr_min_len,
            )
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        except ImportError as e:
            print(
                f"{e}\nInstall retrieval deps:  uv sync --group rag",
                file=sys.stderr,
            )
            sys.exit(1)
        from rag.retrieval import filter_chunks_by_course, make_chunk_retriever

        chunks = filter_chunks_by_course(chunks, course_id)
        if not chunks:
            print(f"No chunks left after filtering course_id={course_id!r}", file=sys.stderr)
            sys.exit(1)
        lecture_ids_in_corpus = {str(c.get("lecture_id") or "") for c in chunks}
        lecture_ids_in_corpus.discard("")
        print(f"Loaded {len(chunks)} chunks, {len(lecture_ids_in_corpus)} distinct lecture_id values")
        try:
            retriever = make_chunk_retriever(
                chunks,
                backend=args.embedder_backend,  # type: ignore[arg-type]
                hf_token=token,
                local_model=args.embedder_model,
                api_model=args.api_embedder_model,
            )
        except ImportError as e:
            print(f"{e}\nInstall: uv sync --group rag   (sentence-transformers)", file=sys.stderr)
            sys.exit(1)

    if args.dry_run:
        return

    from rag.retrieval import (  # noqa: PLC0415
        format_context as fmt_ctx,
        gold_time_interval,
        resolve_lecture_id,
        temporal_hit,
    )

    generation_times: list[float] = []
    retrieval_times: list[float] = []
    per_question: list[dict] = []
    errors: list[tuple[int, str]] = []
    temporal_hits = 0
    unresolved_lecture = 0

    t0 = time.time()
    for i, item in enumerate(items, 1):
        gold = (item.get("answer") or "").strip()
        qtext = (item.get("question") or "").strip()
        gold_iv = gold_time_interval(item)

        resolved_lecture: str | None = None
        context = ""
        ret_s = 0.0
        t_hit = False
        top_chunks: list[dict] = []

        if retriever is not None:
            q_label = str(item.get("lecture_name") or "")
            resolved_lecture = resolve_lecture_id(q_label, lecture_ids_in_corpus)
            if resolved_lecture is None:
                unresolved_lecture += 1
            top_chunks, _, ret_s = retriever.search_timed(
                qtext, k=args.k, lecture_id=resolved_lecture
            )
            retrieval_times.append(ret_s)
            context = fmt_ctx(top_chunks, max_chars=args.max_context_chars)
            t_hit = temporal_hit(top_chunks, gold_iv)
            if t_hit:
                temporal_hits += 1
            user_msg = build_user_message_rag(context, item)
        else:
            retrieval_times.append(0.0)
            user_msg = build_user_message_closed_book(item)

        print(f"\n{'=' * 60}\n  Question {i}/{len(items)}\n{'=' * 60}")
        print(user_msg[:700] + ("…" if len(user_msg) > 700 else ""))
        if retriever is not None:
            print(
                f"\n  resolved_lecture_id={resolved_lecture!r}  "
                f"gold_interval_s={gold_iv}  temporal_hit={t_hit}"
            )

        g0 = time.time()
        try:
            response = call_hf_chat(
                model=args.model,
                system_prompt=system_prompt,
                user_content=user_msg,
                api_key=token,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_s=args.timeout,
            )
        except Exception as e:
            response = ""
            errors.append((i, str(e)))
            print(f"  ERROR: {e}")
        gen_time = time.time() - g0
        generation_times.append(gen_time)

        f1 = f1_score(response, gold)
        em = exact_match_score(response, gold)
        rec = recall(response, gold)
        row: dict = {
            "index": item.get("index", i - 1),
            "lecture_name": item.get("lecture_name"),
            "resolved_lecture_id": resolved_lecture,
            "question": item.get("question"),
            "gold": gold,
            "response": response,
            "f1": f1,
            "exact_match": em,
            "recall": rec,
            "generation_s": gen_time,
            "retrieval_s": ret_s,
            "temporal_hit": t_hit if retriever is not None else None,
            "gold_interval_s": list(gold_iv) if gold_iv else None,
            "answerability_type": item.get("answerability_type"),
            "top_chunk_ids": [c.get("id") for c in top_chunks],
        }
        per_question.append(row)

        print("\n--- Results ---")
        print(f"  Expected:   {gold}")
        print(f"  Response:   {response}")
        if retriever is not None:
            print(f"  Retrieval:  {ret_s:.3f}s")
        print(f"  Generation: {gen_time:.3f}s")
        print(f"  F1 / EM / R: {f1:.3f} / {em:.3f} / {rec:.3f}")

    total_s = time.time() - t0
    n = len(per_question)
    if n == 0:
        print("No questions.")
        return

    avg_f1 = sum(r["f1"] for r in per_question) / n
    avg_em = sum(r["exact_match"] for r in per_question) / n
    avg_rec = sum(r["recall"] for r in per_question) / n
    avg_gen = sum(generation_times) / n
    avg_ret = sum(retrieval_times) / n

    hit_qs = [q for q in per_question if q.get("temporal_hit")]
    miss_qs = [q for q in per_question if not q.get("temporal_hit")]

    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    if retriever is not None:
        print(f"  Temporal retrieval hit: {temporal_hits}/{n} ({100.0 * temporal_hits / n:.1f}%)")
        print(f"  Unresolved lecture_id:  {unresolved_lecture} (fell back to full-corpus search)")
        print(f"  Average retrieval time: {avg_ret:.3f}s")
    print(f"  Average F1:             {avg_f1:.4f}")
    print(f"  Average exact match:    {avg_em:.4f}")
    print(f"  Average recall:         {avg_rec:.4f}")
    print(f"  Average generation:     {avg_gen:.3f}s")
    print(f"  Wall clock total:       {total_s:.1f}s")
    if retriever is not None and (hit_qs or miss_qs):
        print("\n  Conditional QA (gold timestamp overlaps a retrieved chunk window)")
        if hit_qs:
            print(
                f"    Hit (n={len(hit_qs)}):  "
                f"F1={sum(q['f1'] for q in hit_qs)/len(hit_qs):.3f}  "
                f"EM={sum(q['exact_match'] for q in hit_qs)/len(hit_qs):.3f}"
            )
        if miss_qs:
            print(
                f"    Miss (n={len(miss_qs)}): F1={sum(q['f1'] for q in miss_qs)/len(miss_qs):.3f}  "
                f"EM={sum(q['exact_match'] for q in miss_qs)/len(miss_qs):.3f}"
            )
    if errors:
        print(f"  Errors:                 {len(errors)}/{n}")
        for qi, msg in errors[:10]:
            print(f"    Q{qi}: {msg[:200]}")
        if len(errors) > 10:
            print("    …")
    print(f"{'=' * 60}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            meta = {
                "model": args.model,
                "questions_path": str(args.questions),
                "system_prompt_path": str(args.system_prompt_file),
                "retrieval": use_retrieval,
                "chunks_path": str(args.chunks) if use_retrieval else None,
                "build_chunks": args.build_chunks,
                "course_id": course_id,
                "embedder_backend": args.embedder_backend if use_retrieval else None,
                "embedder_model": args.embedder_model if use_retrieval else None,
                "api_embedder_model": args.api_embedder_model if use_retrieval else None,
                "k": args.k if use_retrieval else None,
                "n": n,
                "temporal_hit_rate": temporal_hits / n if use_retrieval else None,
                "avg_f1": avg_f1,
                "avg_exact_match": avg_em,
                "avg_recall": avg_rec,
                "avg_retrieval_s": avg_ret if use_retrieval else None,
                "avg_generation_s": avg_gen,
                "wall_s": total_s,
                "errors": [{"question_index": qi, "message": msg} for qi, msg in errors],
            }
            f.write(json.dumps({"type": "run_summary", **meta}, ensure_ascii=False) + "\n")
            for row in per_question:
                f.write(json.dumps({"type": "question", **row}, ensure_ascii=False) + "\n")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
