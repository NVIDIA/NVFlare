#!/usr/bin/env bash

# NVFLARE INSTALL
NVFLARE_VERSION="2.3.0rc3"
pip install 'nvflare[app_opt]>=${NVFLARE_VERSION}'

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# prepare workspace

workspace="/tmp/workspace"
# clean up to get a fresh restart
if [ -d "${workspace}" ]; then
   rm -r "${workspace}"
fi

# create the workspace directory if not exists
if [ ! -d "${workspace}" ]; then
   mkdir -p ${workspace}
fi

# NVFLARE Provision

# if project.yml file already there use it, otherwise,
# first create project.yml file, then use it to provision

if [ ! -f "${workspace}/project.yml" ]; then
     echo "2" | nvflare provision
     mv "project.yml" ${workspace}/.
fi

server_name="localhost"

python <<END2
from nvflare.lighter.utils import update_project_server_name

project_config_file = "${workspace}/project.yml"
default_server_name = "server1"
server_name = "${server_name}"

update_project_server_name(project_config_file, default_server_name, server_name)
END2

nvflare provision  -p "${workspace}/project.yml" -w ${workspace}

# get last provision project directory
# note the default project name is called "example_project"
# if the project name changed, you need to change here too
project_name="example_project"
prod_dir=$(ls -td ${workspace}/${project_name}/prod_* | head -1)

# update server/local/resources.json
# we decided to update resources.json to set the job-storage and snapshot storage in workspace
# when we rm the workspace, we can remove all the related storages as well.
# in normal production, you don't need to do this.
python <<END1
from nvflare.lighter.utils import update_storage_locations
update_storage_locations(local_dir = "${prod_dir}/${server_name}/local", workspace = "${workspace}")
END1

# start FL system (site-1, site-2 and server)
for s in "site-1" "site-2" $server_name ; do
  startup_dir="${prod_dir}/${s}/startup"
   cmd="${startup_dir}/start.sh"
   eval $cmd
done

# Check if the FL system is ready
python <<END
import os
from nvflare.tool.api_utils import wait_for_system_start
username = "admin@nvidia.com"
prod_dir = "${prod_dir}"
wait_for_system_start(num_clients = 2, prod_dir = prod_dir, username = username, secure_mode = True)
END


