#!/usr/bin/env bash
nvflare simulator job_configs/prostate_central -w ${PWD}/workspaces/prostate_central -c client_All -t 1
nvflare simulator job_configs/prostate_fedavg -w ${PWD}/workspaces/prostate_fedavg -c client_I2CVB,client_MSD,client_NCI_ISBI_3T,client_NCI_ISBI_Dx,client_Promise12,client_PROSTATEx -gpu 0,1,0,1,0,1
nvflare simulator job_configs/prostate_fedprox -w ${PWD}/workspaces/prostate_fedprox -c client_I2CVB,client_MSD,client_NCI_ISBI_3T,client_NCI_ISBI_Dx,client_Promise12,client_PROSTATEx -gpu 0,1,0,1,0,1
nvflare simulator job_configs/prostate_ditto -w ${PWD}/workspaces/prostate_ditto -c client_I2CVB,client_MSD,client_NCI_ISBI_3T,client_NCI_ISBI_Dx,client_Promise12,client_PROSTATEx -gpu 0,1,0,1,0,1

