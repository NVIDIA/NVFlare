#!/usr/bin/env bash
#DOCKER_IMAGE="nvcr.io/nvidia/clara/bionemo-framework:2.3@sha256:715388e62a55ee55f3f0796174576299b121ca9f95d3b83d2b397f7501b21061"
DOCKER_IMAGE="nvcr.io/nvidia/clara/bionemo-framework:nightly"

GPU="all"

NB_DIR="/bionemo_nvflare_examples"

echo "Starting ${DOCKER_IMAGE} with GPU=${GPU}"
echo ""
echo "${COMMAND}"
docker run \
--gpus='"device=4,5,6,7"' --network=host --ipc=host -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
-v ".":/${NB_DIR} \
-w ${NB_DIR} \
${DOCKER_IMAGE} "./start_jupyter.sh"
