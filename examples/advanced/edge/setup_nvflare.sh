PROJECT_NAME=edge_example
PROV_SCRIPT="../../../nvflare/edge/tree_prov.py"

# Check if project directory exists and remove it
if [ -d "$PROJECT_NAME" ]; then
    rm -rf "$PROJECT_NAME"
fi

python "$PROV_SCRIPT" --root_dir ./ -p $PROJECT_NAME -d 1 -w 2

# Define leaf clients array
leaf_clients=(C11 C12 C21 C22)

# Loop through leaf clients and copy resources
for client in "${leaf_clients[@]}"; do
    cp edge__p_resources.json ./$PROJECT_NAME/prod_00/$client/local/
done

