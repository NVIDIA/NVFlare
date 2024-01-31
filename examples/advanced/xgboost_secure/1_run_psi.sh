#!/usr/bin/env bash
echo "Vertical PSI"
nvflare simulator jobs/vertical_xgb_psi -w ${PWD}/workspaces/xgboost_workspace_vertical_psi -n 2 -t 2
