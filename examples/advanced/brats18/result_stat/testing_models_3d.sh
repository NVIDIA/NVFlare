workspace_path="../workspace_brats"
dataset_path="../dataset_brats18/dataset"
datalist_path="../dataset_brats18/datalist"

echo "Centralized"
python3 brats_3d_test_only.py --model_path "${workspace_path}/brats_central/simulate_job/app_server/best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/site-All.json"
echo "FedAvg"
python3 brats_3d_test_only.py --model_path "${workspace_path}/brats_fedavg/simulate_job/app_server/best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/site-All.json"
echo "FedAvgDP"
python3 brats_3d_test_only.py --model_path "${workspace_path}/brats_fedavg_dp/simulate_job/app_server/best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/site-All.json"

