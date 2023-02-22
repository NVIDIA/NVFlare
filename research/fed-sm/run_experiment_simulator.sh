#!/usr/bin/env bash
nvflare simulator job_configs/fedsm_prostate -w ${PWD}/workspaces/fedsm_prostate -c client_I2CVB,client_MSD,client_NCI_ISBI_3T -t 3
