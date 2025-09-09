#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKERFILE=${1:-}
IMAGE_TAG=${2:-}

# Validate inputs
if [[ -z "$DOCKERFILE" || -z "$IMAGE_TAG" ]]; then
  echo "Error: Missing arguments."
  echo "Usage: $0 <Dockerfile> <image_tag>"
  echo "Example: $0 Dockerfile.site nvflare_site"
  exit 1
fi

DOCKERFILE_PATH="$SCRIPT_DIR/$DOCKERFILE"

if [[ ! -f "$DOCKERFILE_PATH" ]]; then
  echo "Error: Dockerfile not found at $DOCKERFILE_PATH"
  exit 1
fi

echo "Calling build_nvflare_docker.sh using Dockerfile $DOCKERFILE and $IMAGE_TAG"
docker build -t "$IMAGE_TAG" -f "$SCRIPT_DIR/$DOCKERFILE" "$SCRIPT_DIR"
docker save "$IMAGE_TAG" | gzip > "$SCRIPT_DIR/${IMAGE_TAG}.tar.gz"

if [[ $? -eq 0 ]]; then
  echo "Docker image successfully saved to: $SCRIPT_DIR/${IMAGE_TAG}.tar.gz"
else
  echo "Failed to save Docker image"
  exit 1
fi
