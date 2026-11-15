#!/usr/bin/env bash
# LIBERO evaluation of a SmolVLA or pi0.5 checkpoint with the paper's protocol: export the checkpoint to a
# lerobot policy folder (LoRA merged into the weights), then run lerobot-eval on the LIBERO simulator with
# 10 episodes per task and replanning every 10 steps.
#
# Usage: bash scripts/eval_libero.sh <config.yaml> <checkpoints/server_round_NNNN.pt>
#   SUITES      comma-separated LIBERO suites (default: all four)
#   N_EPISODES  episodes per task (default: 10)
# Writes the policy folder to outputs/<run>/export/<checkpoint>/ and the results to outputs/<run>/eval/<checkpoint>/,
# so checkpoints of one run can be evaluated at the same time.
set -euo pipefail
cd "$(dirname "$0")/.."

USAGE="usage: bash scripts/eval_libero.sh <config.yaml> <checkpoints/server_round_NNNN.pt>"
CONFIG="${1:?$USAGE}"
CKPT="${2:?$USAGE}"
SUITES="${SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
N_EPISODES="${N_EPISODES:-10}"
N_ACTION_STEPS=10
export MUJOCO_GL="${MUJOCO_GL:-egl}"

RUN_DIR="$(dirname "$(dirname "$CKPT")")"
TAG="$(basename "$CKPT" .pt)"
PRETRAINED=$(python -m src.adapters.lerobot.export \
    --config "$CONFIG" --ckpt "$CKPT" --output-dir "$RUN_DIR/export/$TAG" | tail -1)
echo "[eval] exported policy: $PRETRAINED"

# LIBERO emits observation.images.image (agent view) and image2 (wrist). Map them onto the policy's first two
# cameras, as in training (the generated train_pipeline.yaml); the third camera is zero-filled (empty_cameras=1).
POLICY_TYPE=$(python -c 'import json, sys; print(json.load(open(sys.argv[1] + "/config.json"))["type"])' "$PRETRAINED")
case "$POLICY_TYPE" in
    pi05)
        RENAME_MAP='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
        EXTRA=(--policy.compile_model=false)
        ;;
    smolvla)
        RENAME_MAP='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
        EXTRA=()
        ;;
    *)
        echo "[eval] unsupported policy type: $POLICY_TYPE (expected pi05 or smolvla)" >&2
        exit 2
        ;;
esac

OUT="$RUN_DIR/eval/$TAG"
mkdir -p "$OUT"
echo "[eval] $POLICY_TYPE  ckpt=$CKPT  suites=$SUITES  n_episodes=$N_EPISODES  n_action_steps=$N_ACTION_STEPS"
lerobot-eval \
    --policy.path="$PRETRAINED" \
    --rename_map="$RENAME_MAP" \
    --env.type=libero \
    --env.task="$SUITES" \
    --env.max_parallel_tasks=1 \
    --eval.batch_size=1 \
    --eval.n_episodes="$N_EPISODES" \
    --policy.n_action_steps="$N_ACTION_STEPS" \
    --policy.empty_cameras=1 \
    ${EXTRA[@]+"${EXTRA[@]}"} \
    --output_dir="$OUT"
echo "[eval] results in $OUT"
