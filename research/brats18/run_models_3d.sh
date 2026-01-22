DATASET_ROOT="${PWD}/dataset_brats18/dataset"
DATALIST_ROOT="${PWD}/dataset_brats18/datalist"

python job.py --n_clients 1 --num_rounds 100 --gpu 0 \
  --dataset_base_dir "${DATASET_ROOT}" --datalist_json_path "${DATALIST_ROOT}"

python job.py --n_clients 4 --num_rounds 100 --gpu 0 --threads 4 \
  --dataset_base_dir "${DATASET_ROOT}" --datalist_json_path "${DATALIST_ROOT}"

python job.py --n_clients 4 --num_rounds 100 --gpu 0 --threads 4 --enable_dp \
  --dataset_base_dir "${DATASET_ROOT}" --datalist_json_path "${DATALIST_ROOT}"