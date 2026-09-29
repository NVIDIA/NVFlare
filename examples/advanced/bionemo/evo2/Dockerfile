# syntax=docker/dockerfile:1.4
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
FROM nvcr.io/nvidia/pytorch:26.02-py3@sha256:bbc2b67e2533edd63ff1496bb1ed00a00338cdc1478af6c1a0bf9f4b369977e7

ARG BIONEMO_RECIPES_SHA=101d88b7e0b0ee40e5af303575eb9661620a3706
ARG MEGATRON_BRIDGE_SHA=fcbb6031103d0ca845c1a54d4fee55ecfcca17b6
ARG CAUSAL_CONV1D_SHA=c51519fb10d92ffa257c1982e1831d3243454b2f

COPY --from=ghcr.io/astral-sh/uv:0.12.4@sha256:d0a6eca6c669dc7e9c51218707b8438a3d30402733d739dcc00adb3e213e8f5c \
    /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    bubblewrap \
    git \
    uidmap \
    && rm -rf /var/lib/apt/lists/*

RUN git init /workspace/bionemo \
    && git -C /workspace/bionemo remote add origin https://github.com/NVIDIA-BioNeMo/bionemo-recipes.git \
    && git -C /workspace/bionemo fetch --depth 1 origin "${BIONEMO_RECIPES_SHA}" \
    && git -C /workspace/bionemo checkout --detach FETCH_HEAD \
    && test "$(git -C /workspace/bionemo rev-parse HEAD)" = "${BIONEMO_RECIPES_SHA}"

WORKDIR /workspace/bionemo/recipes/evo2_megatron
ENV UV_LINK_MODE=copy
ENV UV_NO_MANAGED_PYTHON=1
ENV VIRTUAL_ENV=/workspace/bionemo/recipes/evo2_megatron/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    ./.ci_build.sh \
    && python -c \
        'import importlib.metadata as metadata, json, sys; ' \
        'expected = sys.argv[1]; packages = ("megatron-bridge", "megatron-core"); ' \
        'commits = {p: json.loads(metadata.distribution(p).read_text("direct_url.json"))["vcs_info"]["commit_id"] for p in packages}; ' \
        'assert set(commits.values()) == {expected}, f"expected Megatron Bridge commit {expected}, got {commits}"' \
        "${MEGATRON_BRIDGE_SHA}"

# The upstream setup accepts an importable cached causal-conv1d wheel, even if
# that wheel was built against a different NGC Torch ABI. Always rebuild the
# native extension in this image. start_evo2.sh exercises the CUDA ABI after
# Docker has attached the GPU to the running container.
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS=16 python -m pip install \
    --force-reinstall \
    --no-cache-dir \
    --no-deps \
    --no-build-isolation \
    "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@${CAUSAL_CONV1D_SHA}" \
    && python -c "import causal_conv1d"

COPY requirements-runtime.txt /tmp/evo2-requirements-runtime.txt

# NVFlare itself is installed from the repository mounted by start_evo2.sh.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r /tmp/evo2-requirements-runtime.txt

ENV BIONEMO_EVO2_CLASSIFIER=/workspace/bionemo/recipes/evo2_megatron/examples/evo2_classifier.py
WORKDIR /workspace/nvflare/examples/advanced/bionemo/evo2
CMD ["bash"]
