from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Strategy = Literal["speech_anchored", "time_window"]
ChunkPipeline = Literal[
    "cogsci_c127_frames_gpt54_overlap",
    "triple_speech_anchored",
    "triple_time_window",
    "stub_discover_only",
]


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
    course_id: str
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
                "course_id": self.course_id,
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
    *,
    course_id: str = "",
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
        cid = (
            f"{course_id}:{lecture_id}#s{i:05d}"
            if course_id
            else f"{lecture_id}#{i:05d}"
        )
        out.append(
            ChunkRecord(
                id=cid,
                course_id=course_id,
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
    *,
    course_id: str = "",
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
        cid = (
            f"{course_id}:{lecture_id}#w{idx:05d}"
            if course_id
            else f"{lecture_id}#w{idx:05d}"
        )
        out.append(
            ChunkRecord(
                id=cid,
                course_id=course_id,
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


def build_overlapping_frame_window_chunks(
    lecture_id: str,
    frames: list[dict[str, Any]],
    *,
    window_s: float = 45.0,
    overlap_s: float = 15.0,
    frame_min_len: int = 1,
    course_id: str,
    source_label: str = "frame_descriptions_gpt54",
) -> list[ChunkRecord]:
    """
    Sliding time windows over frame-description rows only.
    step = window_s - overlap_s (e.g. 45s window, 15s overlap -> 30s step).
    """
    if overlap_s < 0 or window_s <= 0:
        raise ValueError("window_s must be > 0 and overlap_s must be >= 0")
    if overlap_s >= window_s:
        raise ValueError("overlap_s must be strictly less than window_s")
    step = window_s - overlap_s
    frames = sorted(frames, key=_row_time)
    if not frames:
        return []
    t_end_lecture = max(_row_time(r) for r in frames) + 1.0
    out: list[ChunkRecord] = []
    t = 0.0
    idx = 0
    while t < t_end_lecture:
        t1 = t + window_s
        frame_lines = _lines_in_range(frames, t, t1, min_len=frame_min_len)
        if frame_lines:
            body = "[Frame description]\n" + "\n\n".join(frame_lines)
            cid = f"{course_id}:{lecture_id}#w{idx:05d}"
            out.append(
                ChunkRecord(
                    id=cid,
                    course_id=course_id,
                    lecture_id=lecture_id,
                    t_start=t,
                    t_end=t1,
                    text=body,
                    sources_used=[source_label],
                    index=idx,
                )
            )
            idx += 1
        t += step
    return out


def discover_lecture_groups(corpus_root: Path) -> dict[str, dict[str, Path]]:
    """
    Return lecture_id -> modality -> file path.
    Expects subdirs: frame_descriptions, ocr, speech (legacy lowercase layout).
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


def discover_frame_jsons_by_stem(frames_dir: Path) -> dict[str, Path]:
    """lecture stem -> path for any modality that stores one JSON list per lecture."""
    out: dict[str, Path] = {}
    if not frames_dir.is_dir():
        return out
    for p in frames_dir.glob("*.json"):
        stem = normalize_lecture_stem(p.name)
        out[stem] = p
    return out


def load_lecture_bundle(
    paths: dict[str, Path],
) -> tuple[list[dict], list[dict], list[dict]]:
    frames: list[dict] = _read_json_list(paths["frame_descriptions"])
    ocr: list[dict] = _read_json_list(paths["ocr"])
    speech: list[dict] = _read_json_list(paths["speech"])
    return frames, ocr, speech


@dataclass(frozen=True)
class CourseChunkingSpec:
    """Declarative layout + pipeline for one course."""

    course_id: str
    pipeline: ChunkPipeline
    # Disk layout is read from retrieval_root/<corpus_subdir>/ when set, else course_id.
    corpus_subdir: str | None = None
    # Relative paths inside the course directory (None if unused for that pipeline).
    rel_frame_descriptions_gpt54: str | None = None
    rel_frame_descriptions: str | None = None
    rel_ocr: str | None = None
    rel_speech: str | None = None
    window_s: float | None = None
    overlap_s: float | None = None


# Course registry: extend with new entries as you onboard courses.
# Video is intentionally not referenced here; ingest stays JSON-only.
COURSE_SPECS: dict[str, CourseChunkingSpec] = {
    "cogsci_c127": CourseChunkingSpec(
        course_id="cogsci_c127",
        pipeline="triple_time_window",
        rel_frame_descriptions_gpt54="frame descriptions gpt-5.4",
        rel_frame_descriptions="frame descriptions",
        rel_ocr="OCR",
        rel_speech="speech",
        window_s=45.0,
        overlap_s=15.0,
    ),
    # Stub: Berkeley-style folders with capital OCR; wire a real pipeline when needed.
    "stub_berkeley_style_modalities": CourseChunkingSpec(
        course_id="stub_berkeley_style_modalities",
        pipeline="stub_discover_only",
        corpus_subdir="cogsci_c127",
        rel_frame_descriptions_gpt54="frame descriptions gpt-5.4",
        rel_frame_descriptions="frame descriptions",
        rel_ocr="OCR",
        rel_speech="speech",
    ),
    # Stub: legacy flat lowercase triple used elsewhere in this repo.
    "stub_legacy_lowercase_triple": CourseChunkingSpec(
        course_id="stub_legacy_lowercase_triple",
        pipeline="stub_discover_only",
        rel_frame_descriptions="frame_descriptions",
        rel_ocr="ocr",
        rel_speech="speech",
    ),
}


def resolve_course_spec(course_id: str) -> CourseChunkingSpec:
    if course_id not in COURSE_SPECS:
        known = ", ".join(sorted(COURSE_SPECS))
        raise KeyError(f"Unknown course_id {course_id!r}. Known: {known}")
    return COURSE_SPECS[course_id]


def course_data_dir(retrieval_root: Path, spec: CourseChunkingSpec) -> Path:
    """Directory on disk that holds modality subfolders for this spec."""
    name = spec.corpus_subdir or spec.course_id
    return (retrieval_root / name).resolve()


def course_modal_paths(
    course_dir: Path, spec: CourseChunkingSpec
) -> dict[str, Path | None]:
    """Resolved paths for each modality (None if not configured or missing on disk)."""

    def _p(rel: str | None) -> Path | None:
        if rel is None:
            return None
        path = course_dir / rel
        return path if path.is_dir() else None

    return {
        "frame_descriptions_gpt54": _p(spec.rel_frame_descriptions_gpt54),
        "frame_descriptions": _p(spec.rel_frame_descriptions),
        "ocr": _p(spec.rel_ocr),
        "speech": _p(spec.rel_speech),
    }


def stub_list_sources(course_dir: Path, spec: CourseChunkingSpec) -> dict[str, Any]:
    """
    Introspection helper for stub courses: which modality dirs exist and how many JSON
    lectures each contains (no chunk emission).
    """
    paths = course_modal_paths(course_dir, spec)
    counts: dict[str, int] = {}
    stems_by_mod: dict[str, list[str]] = {}
    for name, p in paths.items():
        if p is None:
            counts[name] = 0
            stems_by_mod[name] = []
            continue
        stems = sorted(discover_frame_jsons_by_stem(p))
        stems_by_mod[name] = stems
        counts[name] = len(stems)
    return {
        "course_id": spec.course_id,
        "paths": paths,
        "json_counts": counts,
        "stems": stems_by_mod,
    }


def _process_cogsci_c127_frames_only(
    course_dir: Path,
    spec: CourseChunkingSpec,
) -> list[dict[str, Any]]:
    gpt_dir = course_modal_paths(course_dir, spec)["frame_descriptions_gpt54"]
    if gpt_dir is None:
        raise FileNotFoundError(
            f"Missing directory for course {spec.course_id!r}: "
            f"{course_dir / (spec.rel_frame_descriptions_gpt54 or '<none>')}"
        )
    w = spec.window_s or 45.0
    ovl = spec.overlap_s if spec.overlap_s is not None else 15.0
    by_stem = discover_frame_jsons_by_stem(gpt_dir)
    rows: list[dict[str, Any]] = []
    for lecture_id in sorted(by_stem):
        frames = _read_json_list(by_stem[lecture_id])
        chunks = build_overlapping_frame_window_chunks(
            lecture_id,
            frames,
            window_s=w,
            overlap_s=ovl,
            course_id=spec.course_id,
            source_label="frame_descriptions_gpt54",
        )
        for c in chunks:
            rows.append(c.to_dict())
    return rows


def _process_triple_pipeline(
    course_dir: Path,
    spec: CourseChunkingSpec,
    strategy: Strategy,
    window_s: float,
    ocr_min_len: int,
) -> list[dict[str, Any]]:
    """Legacy triple (frame_descriptions + ocr + speech) under course_dir."""
    paths_map = course_modal_paths(course_dir, spec)
    fd = paths_map.get("frame_descriptions")
    oc = paths_map.get("ocr")
    sp = paths_map.get("speech")
    if fd is None or oc is None or sp is None:
        raise FileNotFoundError(
            f"Triple pipeline requires frame_descriptions, ocr, and speech dirs under {course_dir}"
        )
    by_stem: dict[str, dict[str, Path]] = {}
    for modality, root in (
        ("frame_descriptions", fd),
        ("ocr", oc),
        ("speech", sp),
    ):
        for stem, p in discover_frame_jsons_by_stem(root).items():
            by_stem.setdefault(stem, {})[modality] = p
    rows: list[dict[str, Any]] = []
    for lecture_id, modalities in sorted(by_stem.items()):
        if not all(k in modalities for k in ("frame_descriptions", "ocr", "speech")):
            continue
        frames, ocr, speech = load_lecture_bundle(modalities)
        if strategy == "speech_anchored":
            chunks = build_speech_anchored_chunks(
                lecture_id,
                speech,
                frames,
                ocr,
                ocr_min_len=ocr_min_len,
                course_id=spec.course_id,
            )
        else:
            chunks = build_time_window_chunks(
                lecture_id,
                speech,
                frames,
                ocr,
                window_s=window_s,
                ocr_min_len=ocr_min_len,
                course_id=spec.course_id,
            )
        for c in chunks:
            rows.append(c.to_dict())
    return rows


def process_course(
    retrieval_root: Path,
    course_id: str,
    *,
    strategy: Strategy = "speech_anchored",
    window_s: float = 60.0,
    ocr_min_len: int = 12,
) -> list[dict[str, Any]]:
    """
    Build chunks for a single course. Retrieval layout::

        retrieval_root/
          <course_id>/
            ... modality JSON dirs ...

    Every chunk dict includes metadata['course_id'] so retrieval can filter to one course.
    """
    spec = resolve_course_spec(course_id)
    course_dir = course_data_dir(retrieval_root, spec)
    if not course_dir.is_dir():
        raise FileNotFoundError(f"Course directory not found: {course_dir}")

    if spec.pipeline == "cogsci_c127_frames_gpt54_overlap":
        return _process_cogsci_c127_frames_only(course_dir, spec)
    if spec.pipeline == "triple_speech_anchored":
        return _process_triple_pipeline(
            course_dir, spec, "speech_anchored", window_s, ocr_min_len
        )
    if spec.pipeline == "triple_time_window":
        return _process_triple_pipeline(
            course_dir, spec, "time_window", window_s, ocr_min_len
        )
    if spec.pipeline == "stub_discover_only":
        # Deliberate no-op for chunk emission; use stub_list_sources() or extend spec.
        _ = stub_list_sources(course_dir, spec)
        return []

    raise NotImplementedError(f"Unhandled pipeline {spec.pipeline!r}")


def process_corpus(
    corpus_root: Path,
    strategy: Strategy = "speech_anchored",
    window_s: float = 60.0,
    ocr_min_len: int = 12,
    *,
    course_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Back-compat entry point.

    If ``course_id`` is set, ``corpus_root`` is treated as ``retrieval_root`` and
    :func:`process_course` is used.

    Otherwise ``corpus_root`` is treated as the **course directory** itself (legacy CLI)
    and the old single-folder triple discovery runs without ``course_id`` in metadata.
    """
    if course_id is not None:
        return process_course(
            corpus_root,
            course_id,
            strategy=strategy,
            window_s=window_s,
            ocr_min_len=ocr_min_len,
        )
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
