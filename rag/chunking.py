from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Strategy = Literal["speech_anchored", "time_window"]


def normalize_lecture_stem(filename: str) -> str:
    """Map 'Lecture13part2_lecture (1).json' -> 'Lecture13part2_lecture'."""
    base = Path(filename).stem
    return re.sub(r"\s*\(\d+\)\s*$", "", base).strip()


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def _row_time(row: dict[str, Any]) -> float:
    v = row.get("timestamp", row.get("t"))
    if v is None:
        raise KeyError("row must have 'timestamp' or 't'")
    return float(v)


def _speech_end_s(row: dict[str, Any], next_row: dict[str, Any] | None) -> float:
    if "end_ms" in row and row["end_ms"] is not None:
        return float(row["end_ms"]) / 1000.0
    if next_row is not None:
        return _row_time(next_row)
    return _row_time(row) + 30.0


_NOISE_OCR_EXACT = frozenset(
    {
        "home",
        "insert",
        "draw",
        "design",
        "transitions",
        "animations",
        "record",
        "review",
        "view",
        "acrobat",
        "comments",
        "share",
        "paste",
        "slides",
        "zoom",
        "file",
        "edit",
        "format",
        "arrange",
        "tools",
        "window",
        "help",
        "powerpoint",
        "autosave",
    }
)


def _is_noise_ocr_line(s: str) -> bool:
    t = s.strip()
    if len(t) < 4:
        return True
    low = t.lower()
    if low in _NOISE_OCR_EXACT and len(t) < 20:
        return True
    if "cmd + ctrl" in low or "cmd+ctrl" in low.replace(" ", ""):
        return True
    if "autosave" in low or "accessibility: investigate" in low:
        return True
    if "english (united states)" in low or "click to add notes" in low:
        return True
    if t.startswith("Tue ") or t.startswith("Mon ") or t.startswith("Wed "):
        return True
    if "saved to my mac" in low:
        return True
    if re.search(r"127Lect|search \(cmd", low):
        return True
    if re.match(r"^Slide \d+ of \d+$", t, re.I):
        return True
    if re.match(r"^\d+%$", t):
        return True
    if len(t) < 6 and t.isalpha() and len(t) <= 2:
        return True
    if re.search(r"\bx x AV\b|Aa v|ab x x", t):
        return True
    if t.endswith("...") and len(t) < 24 and t.count(" ") <= 3:
        return True
    toks = t.split()
    if toks and len(t) < 32 and max(len(w) for w in toks) <= 2:
        return True
    if re.match(r"^([A-Z][\s.]+){2,}.*\.\.\.?$", t) and len(t) < 28:
        return True
    return False


def ocr_text_to_content_lines(ocr_text: str, min_len: int) -> list[str]:
    """Split multiline OCR dumps and drop obvious PowerPoint/Canvas UI lines."""
    out: list[str] = []
    for line in ocr_text.replace("\r\n", "\n").split("\n"):
        s = line.strip()
        if len(s) < min_len:
            continue
        if _is_noise_ocr_line(s):
            continue
        out.append(s)
    return out


def _dedupe_preserve_order(lines: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for s in lines:
        key = s.strip()
        if len(key) < 2:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _ocr_lines_in_range(
    rows: list[dict[str, Any]],
    t0: float,
    t1: float,
    min_len: int,
) -> list[str]:
    acc: list[str] = []
    for r in rows:
        t = _row_time(r)
        if t < t0 or t >= t1:
            continue
        raw = r.get("text") or ""
        for line in ocr_text_to_content_lines(str(raw), min_len=min_len):
            acc.append(line)
    return _dedupe_preserve_order(acc)


def _lines_in_range(
    rows: list[dict[str, Any]],
    t0: float,
    t1: float,
    min_len: int = 1,
) -> list[str]:
    """Collect text from rows with timestamp in [t0, t1), deduplicated in time order."""
    acc: list[str] = []
    for r in rows:
        t = _row_time(r)
        if t < t0 or t >= t1:
            continue
        text = (r.get("text") or "").strip()
        if len(text) < min_len:
            continue
        acc.append(text)
    return _dedupe_preserve_order(acc)


@dataclass
class ChunkRecord:
    id: str
    lecture_id: str
    t_start: float
    t_end: float
    text: str
    sources_used: list[str] = field(default_factory=list)
    index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "lecture_id": self.lecture_id,
            "t_start": round(self.t_start, 4),
            "t_end": round(self.t_end, 4),
            "text": self.text,
            "metadata": {
                "chunk_index": self.index,
                "sources": self.sources_used,
            },
        }


def _compose_chunk_text(
    speech: str,
    ocr: list[str],
    frames: list[str],
    include_empty_modality_headers: bool = False,
) -> tuple[str, list[str]]:
    parts: list[str] = []
    used: list[str] = []
    if speech:
        parts.append(f"[Speech]\n{speech}")
        used.append("speech")
    ocr_text = "\n".join(ocr) if ocr else ""
    if ocr_text or include_empty_modality_headers:
        if ocr_text:
            parts.append(f"[OCR / on-screen]\n{ocr_text}")
            used.append("ocr")
    frame_text = "\n".join(frames) if frames else ""
    if frame_text or include_empty_modality_headers:
        if frame_text:
            parts.append(f"[Frame description]\n{frame_text}")
            used.append("frame_descriptions")
    return "\n\n".join(parts), used


def build_speech_anchored_chunks(
    lecture_id: str,
    speech: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    ocr: list[dict[str, Any]],
    ocr_min_len: int = 12,
) -> list[ChunkRecord]:
    """One chunk per speech segment; attach OCR and frame text in the same time range."""
    speech = sorted(speech, key=_row_time)
    frames = sorted(frames, key=_row_time)
    ocr_rows = sorted(ocr, key=_row_time)
    out: list[ChunkRecord] = []
    for i, row in enumerate(speech):
        t0 = _row_time(row)
        t1 = _speech_end_s(row, speech[i + 1] if i + 1 < len(speech) else None)
        st = (row.get("text") or "").strip()
        ocr_lines = _ocr_lines_in_range(ocr_rows, t0, t1, ocr_min_len)
        frame_lines = _lines_in_range(frames, t0, t1, min_len=20)
        body, used = _compose_chunk_text(st, ocr_lines, frame_lines)
        cid = f"{lecture_id}#{i:05d}"
        out.append(
            ChunkRecord(
                id=cid,
                lecture_id=lecture_id,
                t_start=t0,
                t_end=t1,
                text=body,
                sources_used=used,
                index=i,
            )
        )
    return out


def build_time_window_chunks(
    lecture_id: str,
    speech: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    ocr: list[dict[str, Any]],
    window_s: float = 60.0,
    ocr_min_len: int = 12,
) -> list[ChunkRecord]:
    """Fixed-length windows; concatenate speech, OCR, and frame text in each window."""
    speech = sorted(speech, key=_row_time)
    frames = sorted(frames, key=_row_time)
    ocr_rows = sorted(ocr, key=_row_time)
    if not speech and not frames and not ocr_rows:
        return []
    end_candidates = [0.0]
    for rows in (speech, frames, ocr_rows):
        for r in rows:
            end_candidates.append(_row_time(r))
    t_end_lecture = max(end_candidates) + 1.0
    t = 0.0
    idx = 0
    out: list[ChunkRecord] = []
    while t < t_end_lecture:
        t1 = t + window_s
        speech_bits = _lines_in_range(speech, t, t1, min_len=1)
        speech_block = " ".join(speech_bits).strip()
        ocr_lines = _ocr_lines_in_range(ocr_rows, t, t1, ocr_min_len)
        frame_lines = _lines_in_range(frames, t, t1, min_len=20)
        body, used = _compose_chunk_text(speech_block, ocr_lines, frame_lines)
        if not body.strip():
            t = t1
            continue
        cid = f"{lecture_id}#w{idx:05d}"
        out.append(
            ChunkRecord(
                id=cid,
                lecture_id=lecture_id,
                t_start=t,
                t_end=t1,
                text=body,
                sources_used=used,
                index=idx,
            )
        )
        idx += 1
        t = t1
    return out


def discover_lecture_groups(corpus_root: Path) -> dict[str, dict[str, Path]]:
    """
    Return lecture_id -> modality -> file path.
    Expects subdirs: frame_descriptions, ocr, speech.
    """
    subdirs = {
        "frame_descriptions": corpus_root / "frame_descriptions",
        "ocr": corpus_root / "ocr",
        "speech": corpus_root / "speech",
    }
    by_stem: dict[str, dict[str, Path]] = {}
    for modality, d in subdirs.items():
        if not d.is_dir():
            continue
        for p in d.glob("*.json"):
            stem = normalize_lecture_stem(p.name)
            by_stem.setdefault(stem, {})[modality] = p
    return by_stem


def load_lecture_bundle(paths: dict[str, Path]) -> tuple[list[dict], list[dict], list[dict]]:
    frames: list[dict] = _read_json_list(paths["frame_descriptions"])
    ocr: list[dict] = _read_json_list(paths["ocr"])
    speech: list[dict] = _read_json_list(paths["speech"])
    return frames, ocr, speech


def process_corpus(
    corpus_root: Path,
    strategy: Strategy,
    window_s: float = 60.0,
    ocr_min_len: int = 12,
) -> list[dict[str, Any]]:
    """Load all complete lecture triples and emit chunk dicts."""
    groups = discover_lecture_groups(corpus_root)
    rows: list[dict[str, Any]] = []
    for lecture_id, modalities in sorted(groups.items()):
        if not all(k in modalities for k in ("frame_descriptions", "ocr", "speech")):
            continue
        frames, ocr, speech = load_lecture_bundle(modalities)
        if strategy == "speech_anchored":
            chunks = build_speech_anchored_chunks(
                lecture_id, speech, frames, ocr, ocr_min_len=ocr_min_len
            )
        else:
            chunks = build_time_window_chunks(
                lecture_id,
                speech,
                frames,
                ocr,
                window_s=window_s,
                ocr_min_len=ocr_min_len,
            )
        for c in chunks:
            rows.append(c.to_dict())
    return rows


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
