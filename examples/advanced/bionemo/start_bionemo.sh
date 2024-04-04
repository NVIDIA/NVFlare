#!/usr/bin/env bash
DOCKER_IMAGE="nvcr.io/nvidia/clara/bionemo-framework:1.0"

GPU="all"

NB_DIR="/bionemo_nvflare_examples"
COMMAND="export PYTHONHASHSEED=0; pip install ujson; jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token='' --notebook-dir=${NB_DIR} --NotebookApp.allow_origin='*'"

echo "Starting ${DOCKER_IMAGE} with GPU=${GPU}"
echo ""
echo "${COMMAND}"
docker run \
--gpus="device=${GPU}" --network=host --ipc=host -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
-v ".":/${NB_DIR} \
-w ${NB_DIR} \
${DOCKER_IMAGE} /bin/bash -c "${COMMAND}"
