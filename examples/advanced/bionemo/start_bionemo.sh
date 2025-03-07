#!/usr/bin/env bash
#DOCKER_IMAGE="nvcr.io/nvidia/clara/bionemo-framework:2.4.1"
DOCKER_IMAGE="nvcr.io/nvidia/clara/bionemo-framework:nightly"  # You can also try the nightly build for the latest features but this was not tested.

NB_DIR="/bionemo_nvflare_examples"

echo "Starting ${DOCKER_IMAGE} with all GPUs"
echo ""
echo "${COMMAND}"
docker run \
--gpus='"device=all"' --network=host --ipc=host -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
-v ".":/${NB_DIR} \
-w ${NB_DIR} \
${DOCKER_IMAGE} "./start_jupyter.sh"
