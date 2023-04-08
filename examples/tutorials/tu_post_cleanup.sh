#!/usr/bin/env bash


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

workspace="/tmp/workspace"

if [ ! -d "${workspace}" ]; then
  echo "workspace ${workspace} doesn't exist, quit"
  exit
fi

# get last provision project directory
# note the default project name is called "example_project"
# if the project name changed, you need to change here too
project_name="example_project"

prod_dir=$(ls -td ${workspace}/${project_name}/prod_* | head -1)
server_name="localhost"

# first try gracefully shutdown
python <<END
import os, time
from nvflare.tool.api_utils import shutdown_system

username = "admin@nvidia.com"
prod_dir = "${prod_dir}"
shutdown_system(prod_dir=prod_dir, username = username, secure_mode = True)

END

# now forcefully shutdown
for s in "site-1" "site-2" $server_name ; do
  startup_dir="${prod_dir}/${s}/startup"
  echo "stop $s"
  cmd="echo 'y' | ${startup_dir}/stop_fl.sh"
  eval $cmd
done
