data_dir="./dataset_2D"
out_dir="./datalist_2D"
site_IDs="I2CVB MSD NCI_ISBI_3T NCI_ISBI_Dx Promise12 PROSTATEx"
mkdir ${out_dir}
for site in ${site_IDs}; do
  python3 utils/prepare_data_split.py --mode "folder" --data_dir ${data_dir} --site_name ${site} --out_path ${out_dir}/client_${site}.json
done
