#!/usr/bin/env bash

if [ $# -eq 0 ]
then
  server=localhost
else
  server=$1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $DIR
mkdir -p $DIR/../transfer
python3 -m nvflare.fuel.hci.tools.admin --host ${server} --port 8003 --prompt "> " --with_file_transfer --upload_dir=$DIR/../transfer --download_dir=$DIR/../transfer --with_shell --with_login
