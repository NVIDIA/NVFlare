data_dir="./dataset"
out_dir="./datalists"
mkdir ${out_dir}
for site in I2CVB MSD NCI_ISBI_3T NCI_ISBI_Dx; do
  python3 utils/prepare_data_split.py --data_dir ${data_dir} --site_name ${site} --out_path ${out_dir}
done
