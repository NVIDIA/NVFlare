#!/usr/bin/env bash
DATASET_ROOT=${1}

# Create directory if it doesn't exist
mkdir -p ${DATASET_ROOT}

# Download files if they don't exist
echo "Checking for required data files..."
if [ ! -f "${DATASET_ROOT}/train.csv" ] || [ ! -f "${DATASET_ROOT}/dev.csv" ] || [ ! -f "${DATASET_ROOT}/test.csv" ]; then
    echo "Downloading NCBI Disease dataset from Hugging Face..."
    python3 download_data.py ${DATASET_ROOT}
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to download dataset. Please check error messages above."
        exit 1
    fi
else
    echo "Data files already exist, skipping download."
fi

echo ""
echo "Generating client data splits..."
echo "4-client"
python3 utils/data_split.py --data_path ${DATASET_ROOT} --num_clients 4 --random_seed 0 --site_name_prefix 'site-'
echo "2-client"
python3 utils/data_split.py --data_path ${DATASET_ROOT} --num_clients 2 --random_seed 0 --site_name_prefix 'site-'
