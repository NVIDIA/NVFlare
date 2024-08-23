#!/usr/bin/env bash
echo "Testing baseline_xgboost"
python test_xgboost.py --model_path ./workspaces/xgboost_workspace_centralized/model_centralized.json
echo "Testing xgboost_vertical"
python test_xgboost.py --model_path ./workspaces/xgboost_workspace_vertical/site-1/simulate_job/test.model.json
echo "Testing xgboost_horizontal_histogram"
python test_xgboost.py --model_path ./workspaces/xgboost_workspace_2_histogram/site-1/simulate_job/test.model.json
echo "Testing xgboost_horizontal_cyclic"
python test_xgboost.py --model_path ./workspaces/xgboost_workspace_2_cyclic/server/simulate_job/app_server/xgboost_model.json
echo "Testing xgboost_horizontal_bagging"
python test_xgboost.py --model_path ./workspaces/xgboost_workspace_2_bagging/server/simulate_job/app_server/xgboost_model.json
