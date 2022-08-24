workspace_path="../workspace_brats"
dataset_path="../dataset_brats18/dataset"
datalist_path="../dataset_brats18/datalist"

# Replace the job_ids with the ones from your workspace
job_id_cen="b6c7e274-67f9-402e-8fcd-f54f3cea40a9"
job_id_avg="9b05aa10-c79f-444c-b6b0-2ad7a3538e79"
job_id_avg_dp="c98bdbaf-fdf6-4786-b905-b6a1ba7e398c"

echo "Centralized"
python3 brats_3d_test_only.py --model_path "${workspace_path}/server/${job_id_cen}/app_server/best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/site-All.json"
echo "FedAvg"
python3 brats_3d_test_only.py --model_path "${workspace_path}/server/${job_id_avg}/app_server/best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/site-All.json"
echo "FedAvgDP"
python3 brats_3d_test_only.py --model_path "${workspace_path}/server/${job_id_avg_dp}/app_server/best_FL_global_model.pt" --dataset_base_dir ${dataset_path} --datalist_json_path "${datalist_path}/site-All.json"

