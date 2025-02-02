

#!/bin/bash

setup_type="$1"

parent_dir=$(dirname "$BASH_SOURCE")

# Copy resources.json for server
cp ${parent_dir}/${setup_type}/local_config/server/resources.json /tmp/nvflare/poc/example_project/prod_00/server/local/resources.json

# Copy resources.json for site-1
cp ${parent_dir}/${setup_type}/local_config/site-1/resources.json /tmp/nvflare/poc/example_project/prod_00/site-1/local/resources.json

# Copy resources.json for site-2
cp ${parent_dir}/${setup_type}/local_config/site-2/resources.json /tmp/nvflare/poc/example_project/prod_00/site-2/local/resources.json

# Copy resources.json for site-3
cp ${parent_dir}/${setup_type}/local_config/site-3/resources.json /tmp/nvflare/poc/example_project/prod_00/site-3/local/resources.json

# Copy resources.json for site-4
cp ${parent_dir}/${setup_type}/local_config/site-4/resources.json /tmp/nvflare/poc/example_project/prod_00/site-4/local/resources.json

# Copy resources.json for site-5
cp ${parent_dir}/${setup_type}/local_config/site-5/resources.json /tmp/nvflare/poc/example_project/prod_00/site-5/local/resources.json
