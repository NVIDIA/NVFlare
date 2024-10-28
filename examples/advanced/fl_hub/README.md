# FL Hub (Experimental)

## 1. Create your FL workspaces and start all FL systems

### 1.1 Prepare workspaces
```
cd ./workspaces
for SYSTEM in "t1" "t2a" "t2b"; do
  nvflare provision -p ./${SYSTEM}_project.yml
  cp -r ./workspace/${SYSTEM}_project/prod_00 ./${SYSTEM}_workspace
done
cd ..
```

### 1.2 Adjust hub configs

Modify hub clients:
```
cp -r ./config/site_a/* ./workspaces/t1_workspace/t1_client_a/local/.
cp -r ./config/site_b/* ./workspaces/t1_workspace/t1_client_b/local/.
```

Modify t2 server configs:
```
for SYSTEM in "t2a" "t2b"; do
    T2_SERVER_LOCAL=./workspaces/${SYSTEM}_workspace/localhost/local
    mv ${T2_SERVER_LOCAL}/resources.json.default ${T2_SERVER_LOCAL}/resources.json
    sed -i "s|/tmp/nvflare/snapshot-storage|/tmp/nvflare/snapshot-storage_${SYSTEM}|g" ${T2_SERVER_LOCAL}/resources.json
    sed -i "s|/tmp/nvflare/jobs-storage|/tmp/nvflare/hub/jobs/${SYSTEM}|g" ${T2_SERVER_LOCAL}/resources.json
done
```
Enable only one job at a time:
```
# move resources file for t1 server
SERVER_LOCAL=./workspaces/t1_workspace/localhost/local
mv ${SERVER_LOCAL}/resources.json.default ${SERVER_LOCAL}/resources.json
# set max_jobs value
for SYSTEM in "t1" "t2a" "t2b"; do
    SERVER_LOCAL=./workspaces/${SYSTEM}_workspace/localhost/local
    sed -i 's|"max_jobs": 4|"max_jobs": 1|g' ${SERVER_LOCAL}/resources.json
done
```

### 1.3 Start FL systems

# T1 system with 2 clients (a & b)
```
./start_t1.sh
```

# T2a system with 1 client
```
./start_t2a.sh
```

# T2b system with 2 clients
```
./start_t2b.sh
```

### 2. Submit job

Open admin for hub. Provide admin username: `admin@nvidia.com`
```
./workspaces/t1_workspace/admin@nvidia.com/startup/fl_admin.sh
```

Submit job in console. Replace `[HUB_EXAMPLE]` with your local path of this folder

For a simple example, run
```
submit_job [HUB_EXAMPLE]/jobs/numpy-cross-val
```


### 3. (Optional) Clean-up

Shutdown all FL systems
```
./shutdown_systems.sh
```

Note, you can check if nvflare processes are still running with `ps -asx | grep flare`.

Delete workspaces & temp folders
```
rm -r workspaces/workspace
rm -r workspaces/*_workspace

rm -r /tmp/nvflare
```
