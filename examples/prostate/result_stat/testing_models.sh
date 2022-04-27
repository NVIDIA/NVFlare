#!/usr/bin/env bash

echo "Centralized"
python3 ./result_stat/prostate_test_only.py --model_path "./workspace_prostate/server/run_1/app_server/best_FL_global_model.pt"
echo "FedAvg"
python3 ./result_stat/prostate_test_only.py --model_path "./workspace_prostate/server/run_2/app_server/best_FL_global_model.pt"
echo "FedProx"
python3 ./result_stat/prostate_test_only.py --model_path "./workspace_prostate/server/run_3/app_server/best_FL_global_model.pt"
echo "Ditto"
site_IDs="I2CVB MSD NCI_ISBI_3T NCI_ISBI_Dx Promise12 PROSTATEx"
for site in ${site_IDs}; do
  python3 ./result_stat/prostate_test_only.py --model_path "./workspace_prostate/client_${site}/run_4/app_client_${site}/best_personalized_model.pt" --datalist_json_path "./data_preparation/datalist/client_${site}.json"
done
