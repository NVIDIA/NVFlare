ARG BASE_IMAGE=python:3.12

FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install zip
RUN pip install -U pip
RUN pip install git+https://github.com/NVIDIA/NVFlare.git@main
COPY application_code.zip application_code.zip
RUN nvflare pre-install install application_code.zip

ENTRYPOINT ["/opt/nvflare/startup/sub_start.sh"]

