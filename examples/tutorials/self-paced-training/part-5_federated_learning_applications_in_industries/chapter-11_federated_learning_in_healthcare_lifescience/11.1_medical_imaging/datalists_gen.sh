data_dir="/tmp/nvflare/datasets/MSD"
out_dir="/tmp/nvflare/datasets/MSD/datalist"
mkdir ${out_dir}
python3 utils/prepare_data_split.py --data_dir ${data_dir} --out_path ${out_dir}
