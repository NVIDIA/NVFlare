#!/usr/bin/env bash

docker build -t nvflare-site:latest -f docker/Dockerfile .
docker build -t nvflare-job:latest -f docker/Dockerfile.nvflare-job .
