# merge all json for simulate centralized training, keeping the same training/validation/testing split
out_dir="./datalist"
python utils/merge_two_jsons.py --json_1 ${out_dir}/client_I2CVB.json --json_2 ${out_dir}/client_MSD.json --json_out ${out_dir}/client_All.json
site_IDs="NCI_ISBI_3T NCI_ISBI_Dx"
for i in ${site_IDs}; do
  python utils/merge_two_jsons.py --json_1 ${out_dir}/client_${i}.json --json_2 ${out_dir}/client_All.json --json_out ${out_dir}/client_All.json
done
