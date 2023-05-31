#!/usr/bin/env bash
./workspaces/t1_workspace/t1_client_a/startup/stop_fl.sh <<< "y"
./workspaces/t1_workspace/t1_client_b/startup/stop_fl.sh <<< "y"
./workspaces/t1_workspace/localhost/startup/stop_fl.sh <<< "y"

./workspaces/t2a_workspace/site_a-1/startup/stop_fl.sh <<< "y"
./workspaces/t2a_workspace/localhost/startup/stop_fl.sh <<< "y"

./workspaces/t2b_workspace/site_b-1/startup/stop_fl.sh <<< "y"
./workspaces/t2b_workspace/site_b-2/startup/stop_fl.sh <<< "y"
./workspaces/t2b_workspace/localhost/startup/stop_fl.sh <<< "y"
