FROM python:3.14.5-slim-trixie@sha256:7a500125bc50693f2214e842a621440a1b1b9cbb2188f74ab045d29ed2ea5856

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp/nvflare \
    NVFLARE_HOME=/opt/NVFlare

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        tar \
        zip \
    && rm -rf /var/lib/apt/lists/*

COPY nvflare /opt/NVFlare/nvflare
COPY job_templates /opt/NVFlare/job_templates
COPY MANIFEST.in README.md pyproject.toml setup.cfg setup.py versioneer.py /opt/NVFlare/

WORKDIR /opt/NVFlare

RUN python -m pip install --upgrade "pip==26.1.1" \
    && python -m pip install ".[K8S]" \
    && python -m pip check \
    && mkdir -p /tmp/nvflare /var/tmp/nvflare/workspace /workspace \
    && chgrp -R 0 /opt/NVFlare /tmp/nvflare /var/tmp/nvflare /workspace \
    && chmod -R g=u /opt/NVFlare /tmp/nvflare /var/tmp/nvflare /workspace

USER 1001:0

CMD ["nvflare", "--help"]
