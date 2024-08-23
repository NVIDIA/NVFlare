#!/usr/bin/env bash
echo "Training baseline_xgboost.py"
python baseline_xgboost.py

n=2
for xgb_method in cyclic histogram bagging
do
  echo "Training xgboost_horizontal_${xgb_method}"
  nvflare simulator jobs/${n}_${xgb_method} -w ${PWD}/workspaces/xgboost_workspace_${n}_${xgb_method} -n ${n} -t ${n}
done

echo "Training xgboost_vertical"
echo "Running PSI"
# Create the psi job using the predefined psi_csv template
nvflare config -jt ../../../job_templates/
nvflare job create -j ./jobs/vertical_xgb_psi -w psi_csv -sd ./code/psi \
    -f config_fed_client.conf data_split_path=/tmp/dataset/vertical_xgb_data/site-x/data.csv \
    -force
nvflare simulator jobs/vertical_xgb_psi -w ${PWD}/workspaces/xgboost_workspace_vertical_psi -n 2 -t 2
mkdir -p /tmp/xgboost_vertical_psi
cp -r ${PWD}/workspaces/xgboost_workspace_vertical_psi/site-1/simulate_job/site-1 /tmp/xgboost_vertical_psi
cp -r ${PWD}/workspaces/xgboost_workspace_vertical_psi/site-2/simulate_job/site-2 /tmp/xgboost_vertical_psi

echo "Running vertical_xgb"
# Create the vertical xgb job
nvflare job create -j ./jobs/vertical_xgb -w vertical_xgb -sd ./code/vertical_xgb \
    -f config_fed_client.conf data_split_path=/tmp/dataset/vertical_xgb_data/site-x/data.csv \
    psi_path=/tmp/xgboost_vertical_psi/site-x/psi/intersection.txt train_proportion=0.9 \
    -force
nvflare simulator jobs/vertical_xgb -w ${PWD}/workspaces/xgboost_workspace_vertical -n 2 -t 2
