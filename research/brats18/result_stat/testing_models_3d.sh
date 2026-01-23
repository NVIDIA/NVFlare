workspace_path="/tmp/nvflare/simulation"
dataset_path="../dataset_brats18/dataset"
datalist_path="../dataset_brats18/datalist"

echo "Centralized (brats18_1)"
python3 brats_3d_test_only.py --model_path "${workspace_path}/brats18_1/server/simulate_job/app_server/best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/site-All.json"
echo "FedAvg (brats18_4)"
python3 brats_3d_test_only.py --model_path "${workspace_path}/brats18_4/server/simulate_job/app_server/best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/site-All.json"
echo "FedAvg+DP (brats18_4_dp)"
python3 brats_3d_test_only.py --model_path "${workspace_path}/brats18_4_dp/server/simulate_job/app_server/best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/site-All.json"

