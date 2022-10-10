#!/usr/bin/env bash
export PYTHONPATH="${PYTHONPATH}:/media/ziyuexu/Research/Experiment/NVFlare/prostate_simulator"
echo "PYTHONPATH is ${PYTHONPATH}"
nvflare simulator job_configs/prostate_central -w ${PWD}/workspaces/workspace_central -c client_All -t 1
nvflare simulator job_configs/prostate_fedavg -w ${PWD}/workspaces/workspace_fedavg -c client_I2CVB,client_MSD,client_NCI_ISBI_3T,client_NCI_ISBI_Dx -gpu 0,1,0,1
nvflare simulator job_configs/prostate_fedprox -w ${PWD}/workspaces/workspace_fedprox -c client_I2CVB,client_MSD,client_NCI_ISBI_3T,client_NCI_ISBI_Dx -gpu 0,1,0,1
nvflare simulator job_configs/prostate_ditto -w ${PWD}/workspaces/workspace_ditto -c client_I2CVB,client_MSD,client_NCI_ISBI_3T,client_NCI_ISBI_Dx -gpu 0,1,0,1

