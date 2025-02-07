#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
rm -rf workspace
nvflare provision -p project.yml
WORKSPACE="${DIR}/workspace/fed_policy/prod_00"
cp -r policies/site_a/* $WORKSPACE/site_a/local
cp -r policies/site_b/* $WORKSPACE/site_b/local

for i in {1..5}
do
  cp -r ../../hello-world/hello-numpy-sag/jobs/hello-numpy-sag $WORKSPACE/job$i
  cp -r jobs/job$i/* $WORKSPACE/job$i
done

echo Your workspace is "$WORKSPACE"
