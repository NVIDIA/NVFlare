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
python3 -m nvflare.fuel.hci.tools.admin -m $DIR/.. -s fed_admin.json
