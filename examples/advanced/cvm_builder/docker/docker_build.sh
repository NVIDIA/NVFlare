#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKERFILE=${1:-}
IMAGE_TAG=${2:-}
OUTPUT_ARCHIVE=${3:-}

# Validate inputs
if [[ -z "$DOCKERFILE" || -z "$IMAGE_TAG" || -z "$OUTPUT_ARCHIVE" ]]; then
  echo "Error: Missing arguments."
  echo "Usage: $0 <Dockerfile> <image_tag> <output_application.tar>"
  echo "Example: $0 Dockerfile nvflare-site /srv/nvflare/images/application.tar"
  exit 1
fi

DOCKERFILE_PATH="$SCRIPT_DIR/$DOCKERFILE"

if [[ ! -f "$DOCKERFILE_PATH" ]]; then
  echo "Error: Dockerfile not found at $DOCKERFILE_PATH"
  exit 1
fi

OUTPUT_ARCHIVE="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$OUTPUT_ARCHIVE")"
[[ "$OUTPUT_ARCHIVE" == *.tar ]] || {
  echo "Error: output archive must use the .tar suffix: $OUTPUT_ARCHIVE" >&2
  exit 1
}
mkdir -p -- "$(dirname -- "$OUTPUT_ARCHIVE")"

echo "Building $IMAGE_TAG using $DOCKERFILE"
docker build -t "$IMAGE_TAG" -f "$SCRIPT_DIR/$DOCKERFILE" "$SCRIPT_DIR"
docker save --output "$OUTPUT_ARCHIVE" "$IMAGE_TAG"
echo "Docker image successfully saved to: $OUTPUT_ARCHIVE"
echo "Next: set workload.source.path to $OUTPUT_ARCHIVE and run nvflare provision -p project.yml -w ./workspace"
