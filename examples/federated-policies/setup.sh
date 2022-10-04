#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR
rm -rf workspace
nvflare provision -p project.yml
cp policies/site_a/* workspace/fed_policy/prod_00/site_a/local
cp policies/site_b/* workspace/fed_policy/prod_00/site_b/local
cp -r custom workspace/fed_policy/prod_00/site_a
cp -r custom workspace/fed_policy/prod_00/site_b
cp -r jobs workspace/fed_policy/prod_00
WORKSPACE="$DIR/workspace/fed_policy/prod_00"
echo Your workspace is "$WORKSPACE"
