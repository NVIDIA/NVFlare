#!/usr/bin/env bash

docker build -t nvflare-site:latest -f Dockerfile .
docker build -t nvflare-job:latest -f Dockerfile.nvflare-job .
