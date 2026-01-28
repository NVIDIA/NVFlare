
streaming_to_server="$1"
parent_dir=$(dirname "$BASH_SOURCE")
python ${parent_dir}/edit_site_local_resources.py 5 "${streaming_to_server}" /tmp/nvflare/poc/example_project/prod_00