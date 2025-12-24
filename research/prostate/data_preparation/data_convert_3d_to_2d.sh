data_dir="./dataset"
out_dir="./dataset_2D"
site_IDs="I2CVB MSD NCI_ISBI_3T NCI_ISBI_Dx Promise12 PROSTATEx"
mkdir ${out_dir}
for site in ${site_IDs}; do
  python3 utils/preprocess_3d_to_2d.py --data_dir ${data_dir} --site_name ${site} --out_path ${out_dir}
done
