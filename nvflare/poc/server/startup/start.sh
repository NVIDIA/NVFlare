#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "WORKSPACE set to $DIR/.."
mkdir -p $DIR/../transfer 
$DIR/sub_start.sh $1 $2 &
