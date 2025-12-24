data_dir="./dataset"
out_dir="./datalist"
site_IDs="I2CVB MSD NCI_ISBI_3T NCI_ISBI_Dx"
mkdir ${out_dir}
for site in ${site_IDs}; do
  python3 utils/prepare_data_split.py --mode "file" --data_dir ${data_dir} --site_name ${site} --out_path ${out_dir}/client_${site}.json
done
