#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKERFILE=$1
IMAGE_TAG=$2

echo "Calling build_nvflare_docker.sh using Dockerfile $DOCKERFILE and $IMAGE_TAG"
docker build -t "$IMAGE_TAG" -f "$SCRIPT_DIR/$DOCKERFILE" "$SCRIPT_DIR"
docker save "$IMAGE_TAG" | gzip > "$SCRIPT_DIR/${IMAGE_TAG}.tar.gz"
