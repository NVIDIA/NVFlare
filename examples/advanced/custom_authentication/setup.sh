#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
rm -rf workspace
nvflare poc prepare -i project.yml -c site_a
WORKSPACE="/tmp/nvflare/poc/custom_authentication/prod_00"
cp -r security/server/* $WORKSPACE/server1/local

echo Your workspace is "$WORKSPACE"
