ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:24.07-py3
FROM ${PYTORCH_IMAGE}

ARG NVF_VERSION=main
ENV NVF_BRANCH=${NVF_VERSION}

RUN python3 -m pip install -U pip
RUN python3 -m pip install -U setuptools
RUN python3 -m pip install nvflare

WORKDIR /workspace/
RUN git clone https://github.com/NVIDIA/NVFlare.git --branch ${NVF_BRANCH} --single-branch NVFlare
