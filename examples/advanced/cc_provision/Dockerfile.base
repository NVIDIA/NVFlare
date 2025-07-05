ARG BASE_IMAGE=python:3.10

FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install zip
RUN pip install -U pip

