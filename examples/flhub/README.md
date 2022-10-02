# FL Hub POC

## 1. Set up a virtual environment
```
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
```
(If needed) make all shell scripts executable using
```
find . -name ".sh" -exec chmod +x {} \;
```
initialize virtual environment.
```
source ./virtualenv/set_env.sh
```
install required packages for training
```
pip install --upgrade pip
```

Set path variables and install requirements
```
export NVFLARE_HOME=${PWD}/../..
export HUB_EXAMPLE=${NVFLARE_HOME}/examples/flhub

pip install -r ./virtualenv/requirements.txt
```

## 2. Create your FL workspaces and start all FL systems

### 2.1 Prepare workspaces
```
cd ./workspaces
for SYSTEM in "t1" "t2a" "t2b"; do
  python3 -m nvflare.lighter.provision -p ./${SYSTEM}_project.yml
  cp -r ./workspace/${SYSTEM}_project/prod_00 ./${SYSTEM}_workspace
done
cd ..
```

### 2.2 Adjust hub configs

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

### 2.3 Start FL systems

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

### 3. Submit job

Open admin for hub. Provide admin username: `admin@nvidia.com`
```
./workspaces/t1_workspace/admin@nvidia.com/startup/fl_admin.sh
```

Submit job in console. Replace `[HUB_EXAMPLE]` with your local path of this folder
```
submit_job [HUB_EXAMPLE]/job
```

For a simple example, run
```
submit_job [HUB_EXAMPLE]/job-numpy-sag
```

For a MonaiAlgo example, run
```
submit_job [HUB_EXAMPLE]/job
```

### 4. (Optional) Clean-up

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

## Run admin demo

Open JupyterLab
```
jupyter lab demo.ipynb
```
See [here](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) for installing jupyter lab.
