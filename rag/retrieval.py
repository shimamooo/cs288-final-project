"""Dense retrieval over lecture chunks (JSONL or in-memory dicts from chunking)."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

QUERY_PREFIX_BGE = "Represent this sentence for searching relevant passages: "

EmbedderBackend = Literal["auto", "local", "huggingface"]


def parse_timestamp_to_seconds(s: str) -> float | None:
    """Parse labels like 3:31, 48:19:00 (MM:SS:00), or 1:02:03 (H:M:S)."""
    s = (s or "").strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        nums = [int(float(p)) for p in parts]
    except ValueError:
        return None
    if len(nums) == 1:
        return float(nums[0])
    if len(nums) == 2:
        return float(nums[0] * 60 + nums[1])
    if len(nums) == 3:
        a, b, c = nums
        if a > 23:
            return float(a * 60 + b + (c / 100.0 if 0 < c < 100 else 0.0))
        return float(a * 3600 + b * 60 + c)
    return None


def gold_time_interval(item: dict[str, Any]) -> tuple[float, float] | None:
    t0 = parse_timestamp_to_seconds(str(item.get("Timestamp start", "")))
    t1 = parse_timestamp_to_seconds(str(item.get("Timestamp end", "")))
    if t0 is None or t1 is None:
        return None
    if t1 < t0:
        t0, t1 = t1, t0
    return (t0, t1)


def intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return not (a1 < b0 or b1 < a0)


def resolve_lecture_id(label: str, available: Iterable[str]) -> str | None:
    """
    Map dataset lecture_name (e.g. 'Lecture 1_lecture') to chunk lecture_id
    (e.g. 'Lecture_1_slides' on disk).
    """
    avail_set = set(available)
    if not label or not avail_set:
        return None
    q = label.strip()
    if q in avail_set:
        return q
    q_us = q.replace(" ", "_")
    if q_us in avail_set:
        return q_us
    m = re.match(r"Lecture\s+(\d+)\s*_\s*(lecture|slides)\s*$", q, re.I)
    if m:
        num, kind = m.group(1), m.group(2).lower()
        for suffix in (kind, "slides" if kind == "lecture" else "lecture"):
            cand = f"Lecture_{num}_{suffix}"
            if cand in avail_set:
                return cand
    low_map = {a.lower(): a for a in avail_set}
    if q.lower() in low_map:
        return low_map[q.lower()]
    q_compact = re.sub(r"[\s_]+", "", q.lower())
    for a in avail_set:
        if re.sub(r"[\s_]+", "", a.lower()) == q_compact:
            return a
    return None


def load_chunks_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_chunks_by_course(chunks: list[dict[str, Any]], course_id: str) -> list[dict[str, Any]]:
    if not course_id:
        return chunks
    out: list[dict[str, Any]] = []
    for c in chunks:
        meta = c.get("metadata") or {}
        if meta.get("course_id") == course_id:
            out.append(c)
    return out


def format_context(chunks: list[dict[str, Any]], max_chars: int = 12000) -> str:
    parts: list[str] = []
    n = 0
    for i, c in enumerate(chunks, 1):
        lid = c.get("lecture_id", "")
        ts, te = c.get("t_start"), c.get("t_end")
        head = f"[Passage {i}] lecture_id={lid} t={ts}-{te}s\n"
        body = (c.get("text") or "").strip()
        block = head + body
        if n + len(block) > max_chars:
            remain = max_chars - n - len(head) - 20
            if remain > 80:
                parts.append(head + body[:remain] + "\n[...truncated]")
            break
        parts.append(block)
        n += len(block) + 4
    return "\n\n---\n\n".join(parts)


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (mat / norms).astype(np.float32)


def _l2_normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return (v / n).astype(np.float32)


class _ChunkEmbeddingIndex:
    """Rows of self._emb are L2-normalized; pass normalized query vectors."""

    __slots__ = ("chunks", "_emb", "_by_lecture")

    def __init__(self, chunks: list[dict[str, Any]], emb: np.ndarray) -> None:
        if not chunks:
            raise ValueError("Chunk list is empty")
        if len(chunks) != len(emb):
            raise ValueError("chunks and emb length mismatch")
        self.chunks = chunks
        self._emb = _l2_normalize_rows(emb)
        self._by_lecture: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            lid = str(c.get("lecture_id") or "")
            self._by_lecture.setdefault(lid, []).append(i)

    def search(
        self,
        qv: np.ndarray,
        *,
        k: int,
        lecture_id: str | None,
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        qv = _l2_normalize_vec(qv)
        if lecture_id and self._by_lecture.get(lecture_id):
            idxs = self._by_lecture[lecture_id]
            mat = self._emb[idxs]
            sims = mat @ qv
            order = np.argsort(-sims)[:k]
            sel = [idxs[int(j)] for j in order]
            scores = sims[order]
        else:
            sims = self._emb @ qv
            order = np.argsort(-sims)[:k]
            sel = [int(j) for j in order]
            scores = sims[order]
        hits = [dict(self.chunks[i], _score=float(scores[j])) for j, i in enumerate(sel)]
        return hits, scores


def _parse_hf_feature_response(data: Any) -> np.ndarray:
    """Normalize HF feature-extraction JSON to shape (batch, dim)."""
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(str(data.get("error")))
    if isinstance(data, list) and data and isinstance(data[0], (int, float)):
        return np.asarray([data], dtype=np.float32)
    if isinstance(data, list) and data and isinstance(data[0], list):
        first = data[0]
        if first and isinstance(first[0], list):
            flat = [row[0] if isinstance(row[0], list) else row for row in data]
            return np.asarray(flat, dtype=np.float32)
        return np.asarray(data, dtype=np.float32)
    raise RuntimeError(f"Unexpected embedding response type: {type(data)}")


def hf_api_embed_texts(
    texts: list[str],
    *,
    model_id: str,
    hf_token: str,
    batch_size: int = 8,
    timeout_s: int = 120,
) -> np.ndarray:
    """Call legacy HF Inference API feature extraction (no sentence-transformers install)."""
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    all_rows: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        body = json.dumps({"inputs": batch}).encode("utf-8")
        emb_batch: np.ndarray | None = None
        last_detail = ""
        for attempt in range(4):
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Bearer {hf_token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    data = json.load(resp)
                emb = _parse_hf_feature_response(data)
                if emb.ndim == 1:
                    emb = emb.reshape(1, -1)
                if emb.shape[0] != len(batch):
                    raise RuntimeError(
                        f"Embedding batch size mismatch: got {emb.shape[0]} rows for {len(batch)} inputs"
                    )
                emb_batch = emb.astype(np.float32)
                break
            except urllib.error.HTTPError as e:
                last_detail = e.read().decode("utf-8", errors="replace")
                if e.code in (502, 503) and attempt < 3:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"HTTP {e.code}: {last_detail}") from e
        if emb_batch is None:
            raise RuntimeError(f"Embedding failed after retries: {last_detail}")
        all_rows.append(emb_batch)
        time.sleep(0.05)
    return np.vstack(all_rows) if all_rows else np.zeros((0, 0), dtype=np.float32)


class DenseChunkRetriever:
    """Cosine similarity with local sentence-transformers (BGE-style query prefix)."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        embedder_model: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        texts = [(c.get("text") or "").strip() or "(empty)" for c in chunks]
        model = SentenceTransformer(embedder_model)
        embs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=False,
            show_progress_bar=len(texts) > 256,
        )
        self._index = _ChunkEmbeddingIndex(chunks, np.asarray(embs, dtype=np.float32))
        self._model = model
        self.chunks = chunks

    def search(
        self,
        query: str,
        *,
        k: int,
        lecture_id: str | None,
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        qv = np.asarray(
            self._model.encode(
                [QUERY_PREFIX_BGE + query],
                normalize_embeddings=False,
            )[0],
            dtype=np.float32,
        )
        return self._index.search(qv, k=k, lecture_id=lecture_id)

    def search_timed(
        self,
        query: str,
        *,
        k: int,
        lecture_id: str | None,
    ) -> tuple[list[dict[str, Any]], np.ndarray, float]:
        t0 = time.perf_counter()
        hits, scores = self.search(query, k=k, lecture_id=lecture_id)
        return hits, scores, time.perf_counter() - t0


class HubApiDenseChunkRetriever:
    """Same interface as DenseChunkRetriever; embeddings via HF Inference API (urllib only)."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        *,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        hf_token: str,
        batch_size: int = 8,
        timeout_s: int = 120,
    ) -> None:
        if not hf_token.strip():
            raise ValueError("hf_token is required for HubApiDenseChunkRetriever")
        texts = [(c.get("text") or "").strip() or "(empty)" for c in chunks]
        embs = hf_api_embed_texts(
            texts,
            model_id=model_id,
            hf_token=hf_token,
            batch_size=batch_size,
            timeout_s=timeout_s,
        )
        self._index = _ChunkEmbeddingIndex(chunks, embs)
        self._model_id = model_id
        self._hf_token = hf_token
        self._timeout_s = timeout_s
        self.chunks = chunks

    def search(
        self,
        query: str,
        *,
        k: int,
        lecture_id: str | None,
    ) -> tuple[list[dict[str, Any]], np.ndarray]:
        qv = hf_api_embed_texts(
            [query],
            model_id=self._model_id,
            hf_token=self._hf_token,
            batch_size=1,
            timeout_s=self._timeout_s,
        )[0]
        return self._index.search(qv, k=k, lecture_id=lecture_id)

    def search_timed(
        self,
        query: str,
        *,
        k: int,
        lecture_id: str | None,
    ) -> tuple[list[dict[str, Any]], np.ndarray, float]:
        t0 = time.perf_counter()
        hits, scores = self.search(query, k=k, lecture_id=lecture_id)
        return hits, scores, time.perf_counter() - t0


def make_chunk_retriever(
    chunks: list[dict[str, Any]],
    *,
    backend: EmbedderBackend = "auto",
    hf_token: str,
    local_model: str = "BAAI/bge-small-en-v1.5",
    api_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    local_batch_size: int = 32,
    api_batch_size: int = 8,
) -> DenseChunkRetriever | HubApiDenseChunkRetriever:
    if backend == "auto":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            backend = "huggingface"
        else:
            backend = "local"
    if backend == "local":
        return DenseChunkRetriever(chunks, embedder_model=local_model, batch_size=local_batch_size)
    if backend == "huggingface":
        return HubApiDenseChunkRetriever(
            chunks,
            model_id=api_model,
            hf_token=hf_token,
            batch_size=api_batch_size,
        )
    raise ValueError(f"Unknown embedder backend: {backend!r}")


def temporal_hit(
    retrieved: list[dict[str, Any]],
    gold: tuple[float, float] | None,
) -> bool:
    if gold is None:
        return False
    g0, g1 = gold
    for c in retrieved:
        try:
            ts = float(c.get("t_start", 0.0))
            te = float(c.get("t_end", 0.0))
        except (TypeError, ValueError):
            continue
        if intervals_overlap(ts, te, g0, g1):
            return True
    return False
