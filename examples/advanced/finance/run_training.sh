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
nvflare simulator jobs/vertical_xgb_psi -w ${PWD}/workspaces/xgboost_workspace_vertical_psi -n 2 -t 2
mkdir -p /tmp/xgboost_vertical_psi
cp -r ${PWD}/workspaces/xgboost_workspace_vertical_psi/simulate_job/site-* /tmp/xgboost_vertical_psi
nvflare simulator jobs/vertical_xgb -w ${PWD}/workspaces/xgboost_workspace_vertical -n 2 -t 2