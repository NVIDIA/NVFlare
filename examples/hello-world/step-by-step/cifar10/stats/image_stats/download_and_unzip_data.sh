#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATASET_PATH="/tmp/nvflare/image_stats/data"

if [ ! -d $DATASET_PATH ]; then
  mkdir -p $DATASET_PATH
fi

source_url="$1"
echo "download url = ${source_url}"
if [ -n "${source_url}" ]; then
  if [ ! -f "${DATASET_PATH}/COVID-19_Radiography_Dataset.zip" ]; then
    wget -O "${DATASET_PATH}/COVID-19_Radiography_Dataset.zip" "${source_url}"
  else
    echo "zip file exists."
  fi
  if [ ! -d "${DATASET_PATH}/COVID-19_Radiography_Dataset" ]; then
    unzip -d $DATASET_PATH "${DATASET_PATH}/COVID-19_Radiography_Dataset.zip"
  else
    echo "image files exist."
  fi
else
  echo "empty URL, nothing downloaded, you need to provide real URL to download"
fi