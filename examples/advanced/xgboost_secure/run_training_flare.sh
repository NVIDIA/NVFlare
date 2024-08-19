#!/usr/bin/env bash
WORKSPACE_ROOT="/tmp/nvflare/xgb_workspaces"
n=3

echo "Training horizontal"
nvflare simulator jobs/xgb_hori -w ${WORKSPACE_ROOT}/workspace_hori -n ${n} -t ${n}
echo "Prepare secure horizontal tenseal context"
nvflare provision -p project.yml -w ${WORKSPACE_ROOT}/workspace_hori_secure
echo "Training secure horizontal"
nvflare simulator jobs/xgb_hori_secure \ 
    -w ${WORKSPACE_ROOT}/workspace_hori_secure/example_project/prod_00/site-1 -n ${n} -t ${n}
echo "Training vertical"
nvflare simulator jobs/xgb_vert -w ${WORKSPACE_ROOT}/workspace_vert -n ${n} -t ${n}
echo "Training secure vertical"
nvflare simulator jobs/xgb_vert_secure -w ${WORKSPACE_ROOT}/workspace_vert_secure -n ${n} -t ${n}
