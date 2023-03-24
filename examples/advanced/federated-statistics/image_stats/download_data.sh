#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


mkdir -p /tmp/nvflare/image_stats/data

source_url="$1"
echo "download url = ${source_url}"
if [ ! -z "${source_url}" ]; then
  if [ ! -f "/tmp/nvflare/image_stats/data/COVID-19_Radiography_Dataset.zip" ]; then
    wget -O /tmp/nvflare/image_stats/data/COVID-19_Radiography_Dataset.zip "${source_url}"
  else
    echo "file exists."
  fi
  if [ ! -d "/tmp/nvflare/image_stats/data/COVID-19_Radiography_Dataset" ]; then
    unzip -d /tmp/nvflare/image_stats/data /tmp/nvflare/image_stats/data/COVID-19_Radiography_Dataset.zip
  else
    echo 'A' | unzip -d /tmp/nvflare/image_stats/data /tmp/nvflare/image_stats/data/COVID-19_Radiography_Dataset.zip
  fi

else
  echo "nothing download, you need to provide real URL to download"
fi