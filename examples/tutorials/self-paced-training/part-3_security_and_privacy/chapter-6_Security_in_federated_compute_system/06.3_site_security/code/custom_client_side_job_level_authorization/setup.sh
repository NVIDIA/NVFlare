#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
rm -rf workspace
nvflare config -pw /tmp/nvflare/poc
nvflare poc prepare -i project.yml -c site_a
WORKSPACE="/tmp/nvflare/poc/job-level-authorization/prod_00"
cp -r security/site_a/* $WORKSPACE/site_a/local

for i in {1..2}
do
  cp -r ../../hello-world/hello-numpy-sag/jobs/hello-numpy-sag $WORKSPACE/job$i
  cp -r jobs/job$i/* $WORKSPACE/job$i
done

echo Your workspace is "$WORKSPACE"
