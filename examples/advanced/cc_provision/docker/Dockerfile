ARG BASE_IMAGE=python:3.12

FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

RUN pip install -U pip &&\
    pip install nvflare~=2.7.0
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY code/ /local/custom

ENTRYPOINT ["/user_config/nvflare/startup/sub_start.sh", "--verify"]

