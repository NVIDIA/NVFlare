#!/usr/bin/env bash
# Download the four LIBERO suites in GR00T's LeRobot v2 format (IPEC-COMMUNITY/*_no_noops_1.0.0_lerobot,
# public on Hugging Face) and prepare them as Isaac-GR00T's LIBERO example describes
# (third_party/Isaac-GR00T/examples/LIBERO/README.md): add modality.json, generate meta/stats.json and
# patch the one corrupted wrist video in the goal suite. About 2 GB in total.
#
# Usage: bash scripts/prep_groot_libero.sh [data dir]   (default data/groot-libero; run after setup_groot.sh)
set -euo pipefail
cd "$(dirname "$0")/.."

GROOT_DIR="$PWD/third_party/Isaac-GR00T"
DATA="${1:-data/groot-libero}"
PY="$PWD/.venv-groot/bin/python"
mkdir -p "$DATA"
DATA="$(cd "$DATA" && pwd)"

for suite in spatial object goal 10; do
    name="libero_${suite}_no_noops_1.0.0_lerobot"
    dst="$DATA/$name"
    echo "[prep] downloading $name -> $dst"
    "$PY" - "$name" "$dst" <<'PY'
import sys
import time

from huggingface_hub import snapshot_download

name, dst = sys.argv[1], sys.argv[2]
for attempt in range(1, 9):
    try:
        snapshot_download(f"IPEC-COMMUNITY/{name}", repo_type="dataset", local_dir=dst, max_workers=4)
        break
    except Exception as e:  # retry transient HTTP errors (429/5xx); snapshot_download resumes
        wait = min(60, 5 * attempt)
        print(f"  attempt {attempt} failed: {type(e).__name__}: {str(e)[:120]}; retry in {wait}s", flush=True)
        time.sleep(wait)
else:
    raise SystemExit(f"failed to download {name}")
PY
    cp "$GROOT_DIR/examples/LIBERO/modality.json" "$dst/meta/"
    (cd "$GROOT_DIR" && "$PY" gr00t/data/stats.py --dataset-path "$dst" --embodiment-tag LIBERO_PANDA)
done

patch="$GROOT_DIR/examples/LIBERO/patches/episode_000082.mp4"
wrist="$DATA/libero_goal_no_noops_1.0.0_lerobot/videos/chunk-000/observation.images.wrist_image"
if [ -f "$patch" ] && [ -d "$wrist" ]; then
    cp "$patch" "$wrist/"
fi
du -sh "$DATA"/*
echo "[prep] done: $DATA"
