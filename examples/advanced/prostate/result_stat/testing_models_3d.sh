workspace_path="../prostate_3D/workspaces"
dataset_path="../data_preparation/dataset"
datalist_path="../data_preparation/datalist"
echo "Centralized"
python3 prostate_3d_test_only.py --model_path "${workspace_path}/prostate_central/simulate_job/app_server/best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/client_All.json"
echo "FedAvg"
python3 prostate_3d_test_only.py --model_path "${workspace_path}/prostate_fedavg/simulate_job/app_server/best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/client_All.json"
echo "FedProx"
python3 prostate_3d_test_only.py --model_path "${workspace_path}/prostate_fedprox/simulate_job/app_server/best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/client_All.json"
echo "Ditto"
site_IDs="I2CVB MSD NCI_ISBI_3T NCI_ISBI_Dx"
for site in ${site_IDs}; do
  python3 prostate_3d_test_only.py --model_path "${workspace_path}/prostate_ditto/simulate_job/app_client_${site}/best_personalized_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/client_${site}.json"
done
