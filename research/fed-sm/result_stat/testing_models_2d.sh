workspace_path="../workspaces"
dataset_path="DATASET_ROOT/dataset_2D"
datalist_path="DATASET_ROOT/datalist_2D"

echo "FedSM"
python3 prostate_2d_test_only.py --model_path "${workspace_path}/fedsm_prostate/simulate_job/app_server/global_best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/client_All.json"
