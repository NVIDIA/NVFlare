#!/usr/bin/env bash

# NVFLARE INSTALL
NVFLARE_VERSION="2.3.0rc3"
pip install 'nvflare>=${NVFLARE_VERSION}'

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# prepare workspace

workspace="/tmp/workspace"
# clean up to get a fresh restart
if [ -f "${workspace}" ]; then
   rm -r workspace
fi

# create the workspace directory if not exists
if [ ! -f "${workspace}" ]; then
   mkdir -p ${workspace}
fi

# NVFLARE Provision

# if project.yml file already there use it, otherwise,
# first create project.yml file, then use it to provision

if [ ! -f "${workspace}/project.yml" ]; then
     echo "2" | nvflare provision
     mv "project.yml" ${workspace}/.
fi

nvflare provision  -p "${workspace}/project.yml" -w ${workspace}

# get last provision project directory
# note the default project name is called "example_project"
# if the project name changed, you need to change here too
project_name="example_project"

prod_dir=$( ls -td ${workspace}/${project_name}/*/  | head -1)

server_startup_dir="${prod_dir}/server1/startup"
site_1_startup_dir="${prod_dir}/site-1/startup"
site_2_startup_dir="${prod_dir}/site-2/startup"

for s in $server_startup_dir $site_1_startup_dir $site_2_startup_dir ; do
   cmd="${s}/start.sh"
   eval $cmd
done

# Check if the FL system is ready
sleep 20

python <<END

import os, time
from nvflare.fuel.flare_api.flare_api import new_secure_session
from nvflare.fuel.flare_api.flare_api import NoConnection

project_name = "${project_name}"
username = "admin@nvidia.com"
workspace_root = "${workspace}"
prod_dir = "${prod_dir}"

admin_user_dir = os.path.join(workspace_root, project_name, prod_dir, username)

# just in case try to connect before server started

flare_not_ready = True
while flare_not_ready:
    print("trying to connect to server")

    sess = new_secure_session(
        username=username,
        startup_kit_location=admin_user_dir
    )

    sys_info = sess.get_system_info()

    print(f"Server info:\n{sys_info.server_info}")
    print("\nClient info")
    for client in sys_info.client_info:
        print(client)
    flare_not_ready = len( sys_info.client_info) < 2

    time.sleep(2)

if flare_not_ready:
   raise RuntimeError("can't not connect to server")
else:
   print("ready to go")

END


