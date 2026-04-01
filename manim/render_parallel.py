"""
Benchmark: sequential vs parallel rendering of split Manim scenes.

Usage:
    uv run python manim/render_parallel.py [--locale zh] [--quality l]

The script:
  1. Renders Scene1-4 sequentially  →  records wall-clock time
  2. Renders Scene1-4 in parallel   →  records wall-clock time
  3. Stitches the parallel outputs into a single video via ffmpeg
  4. Prints the speedup ratio

NOTE: Run once first so ElevenLabs audio is cached; otherwise sequential
gets a double penalty (generates audio AND renders) making the comparison unfair.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SCENES = ["Scene1", "Scene2", "Scene3", "Scene4"]
SCENES_FILE = "manim/scenes.py"
DOMAIN = "translation-example"

# Manim output subdirectory name for each quality flag
QUALITY_DIR = {"l": "480p15", "m": "720p30", "h": "1080p60", "p": "480p15", "k": "2160p60"}


def ensure_mo(locale: str) -> None:
    """Compile .po → .mo if needed (replicates what manim_render_translation does)."""
    po = Path("locale") / locale / "LC_MESSAGES" / f"{DOMAIN}.po"
    mo = Path("locale") / locale / "LC_MESSAGES" / f"{DOMAIN}.mo"
    if not mo.exists():
        subprocess.run(["msgfmt", str(po), "-o", str(mo)], check=True)


def build_sequential_cmd(scene: str, locale: str, quality: str) -> list[str]:
    """Uses manim_render_translation (fine for sequential — no shared-dir conflicts)."""
    return [
        "uv", "run", "manim_render_translation",
        SCENES_FILE, "-s", scene, "-d", DOMAIN, "-l", locale, "-q", quality,
    ]


def build_parallel_cmd(scene: str, locale: str, quality: str) -> tuple[list[str], dict]:
    """
    Calls manim directly with an isolated --media_dir per scene.
    This prevents delete_nonsvg_files() in one process from deleting .dvi
    files that another process is still using.
    """
    env = os.environ.copy()
    env["LOCALE"] = locale
    env["DOMAIN"] = DOMAIN
    cmd = [
        "uv", "run", "manim",
        f"-q{quality}",
        SCENES_FILE,
        scene,
        "-o", f"{scene}_{locale}.mp4",
        "--disable_caching",
        "--media_dir", f"media/{scene}",
    ]
    return cmd, env


def run_sequential(locale: str, quality: str) -> float:
    print("\n── Sequential ──────────────────────────────")
    start = time.perf_counter()
    for scene in SCENES:
        print(f"  Rendering {scene}...")
        result = subprocess.run(build_sequential_cmd(scene, locale, quality))
        if result.returncode != 0:
            print(f"  ERROR: {scene} failed (exit {result.returncode})")
            sys.exit(result.returncode)
    elapsed = time.perf_counter() - start
    print(f"  Done — {elapsed:.1f}s")
    return elapsed


def pre_warm() -> None:
    """Pre-download theme so parallel processes don't race to download it."""
    try:
        from manim_themes.manim_theme import load_iterm2_theme
        load_iterm2_theme("Andromeda")
        print("  Theme pre-warmed.")
    except Exception as e:
        print(f"  Warning: could not pre-warm theme: {e}")


def run_parallel(locale: str, quality: str) -> float:
    print("\n── Parallel ────────────────────────────────")
    ensure_mo(locale)
    pre_warm()
    start = time.perf_counter()
    procs = {}
    for scene in SCENES:
        print(f"  Spawning {scene}...")
        cmd, env = build_parallel_cmd(scene, locale, quality)
        procs[scene] = subprocess.Popen(cmd, env=env)

    failed = []
    for scene, proc in procs.items():
        rc = proc.wait()
        if rc != 0:
            failed.append(scene)
        print(f"  {scene} finished (exit {rc})")

    elapsed = time.perf_counter() - start
    print(f"  Done — {elapsed:.1f}s")

    if failed:
        print(f"  WARNING: these scenes failed: {failed}")

    return elapsed


def find_output(scene: str, locale: str) -> Path | None:
    """Glob for {scene}_{locale}.mp4 anywhere under media/."""
    candidates = sorted(
        Path("media").glob(f"**/{scene}_{locale}.mp4"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def stitch(locale: str, output: str) -> None:
    parts = []
    for scene in SCENES:
        path = find_output(scene, locale)
        if path is None:
            print(f"  WARNING: output for {scene} not found — skipping stitch")
            return
        parts.append(path)

    concat_file = Path("media") / "concat_list.txt"
    concat_file.write_text("\n".join(f"file '{p.resolve()}'" for p in parts))

    print(f"\n── Stitching → {output} ────────────────────")
    for p in parts:
        print(f"  + {p}")

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", output],
        check=True,
    )
    concat_file.unlink()
    print(f"  Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark parallel vs sequential Manim rendering")
    parser.add_argument("--locale",  "-l", default="zh")
    parser.add_argument("--quality", "-q", default="l", choices=["l", "m", "h", "p", "k"])
    parser.add_argument(
        "--mode",
        choices=["sequential", "parallel", "both"],
        default="both",
        help="Which benchmark to run (default: both)",
    )
    parser.add_argument("--output", "-o", default="output_stitched.mp4")
    args = parser.parse_args()

    seq_time = par_time = None

    if args.mode in ("sequential", "both"):
        seq_time = run_sequential(args.locale, args.quality)

    if args.mode in ("parallel", "both"):
        par_time = run_parallel(args.locale, args.quality)

    print("\n── Results ─────────────────────────────────")
    if seq_time is not None:
        print(f"  Sequential : {seq_time:6.1f}s")
    if par_time is not None:
        print(f"  Parallel   : {par_time:6.1f}s")
    if seq_time and par_time:
        print(f"  Speedup    : {seq_time / par_time:.2f}x")

    if args.mode in ("parallel", "both"):
        stitch(args.locale, args.output)


if __name__ == "__main__":
    main()
