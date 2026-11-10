ARG BASE_IMAGE=python:3.12

FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/local/custom

COPY nvflare_wheels/*.whl /tmp/nvflare_wheels/
RUN python -m pip install /tmp/nvflare_wheels/*.whl
COPY code/ /local/custom
COPY requirements.txt .
RUN pip install -r requirements.txt

ENTRYPOINT ["/user_config/nvflare/startup/sub_start.sh", "--once", "--verify"]
