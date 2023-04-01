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
from nvflare.fuel.flare_api.flare_api import new_secure_session
from nvflare.fuel.flare_api.flare_api import NoConnection
from nvflare.lighter.utils import shutdown_fl_system

username = "admin@nvidia.com"
prod_dir = "${prod_dir}"

admin_user_dir = os.path.join(prod_dir, username)
sess = new_secure_session(username=username, startup_kit_location=admin_user_dir)
shutdown_fl_system(sess)

END

# now forcefully shutdown
for s in "site-1" "site-2" $server_name ; do
  startup_dir="${prod_dir}/${s}/startup"
  echo "stop $s"
  cmd="echo 'y' | ${startup_dir}/stop_fl.sh"
  eval $cmd
done

# remove workspace
if [ -d "${workspace}" ]; then
  echo "wait for 30 seconds, then remove ${workspace}"
  sleep 30
  rm -r ${workspace}
fi