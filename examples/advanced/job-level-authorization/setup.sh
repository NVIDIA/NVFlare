#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
rm -rf workspace
nvflare config -pw /tmp/nvflare/poc
nvflare poc prepare -i project.yml -c site_a
WORKSPACE="/tmp/nvflare/poc/job-level-authorization/prod_00"
cp -r security/site_a/* $WORKSPACE/site_a/local

echo Your workspace is "$WORKSPACE"
