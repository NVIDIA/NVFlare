workspace_path="../prostate_2D/workspace_prostate"
dataset_path="../data_preparation/dataset_2D"
datalist_path="../data_preparation/datalist_2D"
echo "Centralized"
python3 prostate_2d_test_only.py --model_path "${workspace_path}/server/${job_id_cen}/app_server/best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/client_All.json"
echo "FedAvg"
python3 prostate_2d_test_only.py --model_path "${workspace_path}/server/${job_id_avg}/app_server/best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/client_All.json"
echo "FedProx"
python3 prostate_2d_test_only.py --model_path "${workspace_path}/server/${job_id_prox}/app_server/best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/client_All.json"
echo "Ditto"
site_IDs="I2CVB MSD NCI_ISBI_3T NCI_ISBI_Dx Promise12 PROSTATEx"
for site in ${site_IDs}; do
  python3 prostate_2d_test_only.py --model_path "${workspace_path}/client_${site}/${job_id_dit}/app_client_${site}/best_personalized_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/client_${site}.json"
done
