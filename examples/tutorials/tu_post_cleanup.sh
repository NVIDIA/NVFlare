#!/usr/bin/env bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

workspace="/tmp/workspace"

# get last provision project directory
# note the default project name is called "example_project"
# if the project name changed, you need to change here too
project_name="example_project"

prod_dir=$(ls -td ${workspace}/${project_name}/prod_* | head -1)
server_name="localhost"

server_startup_dir="${prod_dir}/${server_name}/startup"
site_1_startup_dir="${prod_dir}/site-1/startup"
site_2_startup_dir="${prod_dir}/site-2/startup"

for s in $site_1_startup_dir $site_2_startup_dir $server_startup_dir ; do
   cmd="echo 'y' | ${s}/stop_fl.sh"
   eval $cmd
done

# remove workspace
if [ -f "${workspace}" ]; then
  echo "removing ${workspace}"
  rm -r "${workspace}"
fi