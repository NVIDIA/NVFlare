#!/usr/bin/env bash
DOCKER_IMAGE="nvcr.io/nvidia/clara/bionemo-framework:1.8"

GPU="all"

NB_DIR="/bionemo_nvflare_examples"

echo "Starting ${DOCKER_IMAGE} with GPU=${GPU}"
echo ""
echo "${COMMAND}"
docker run \
--gpus="device=${GPU}" --network=host --ipc=host -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
-v ".":/${NB_DIR} \
-w ${NB_DIR} \
${DOCKER_IMAGE} "./start_jupyter.sh"
