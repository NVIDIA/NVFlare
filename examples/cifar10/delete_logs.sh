#!/usr/bin/env bash

workspace="workspaces/$1"

if test -z "$1"
then
      echo "Usage: ./delete_logs.sh [workspace], e.g. ./delete_run.sh poc_workspace"
      exit 1
fi

rm ./${workspace}/*/log.txt
