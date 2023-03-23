#!/usr/bin/env bash

nvflare poc --stop
sleep 3

if [ -f "/tmp/nvflare/jobs-storage" ]; then
  rm -r "/tmp/nvflare/jobs-storage"
fi

if [ -f "/tmp/nvflare/snapshot-storage" ]; then
  rm -r "/tmp/nvflare/snapshot-storage"
fi
