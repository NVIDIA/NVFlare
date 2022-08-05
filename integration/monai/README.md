# MONAI Integration

## Objective
Integration with [MONAI](https://monai.io/)'s federated learning capabilities.

Add `Executor` class to allow using [MONAI bundle](https://docs.monai.io/en/latest/bundle.html) configuration files with MONAI's `ClientAlgo` class.

### Goals:

Allow the use of bundles from the MONAI [model zoo](https://github.com/Project-MONAI/model-zoo) or custom configurations with NVFlare.

### Non-goals:

n/a

## Background
MONAI allows the definition of AI models using the "bundle" concept. 
It allows for easy experimentation and sharing of models that have been developed using MONAI.
Using the bundle configurations, we can use MONAI's `MonaiAlgo` class to execute a bundle model in a federated scenario using NVFlare.

NVFlare executes the `MonaiAlgo` class using the `ClientAlgoExecutor` class provided here.

## Description
In the following, we show an example of running MONAI-bundle configurations with NVFlare.

### (Optional) 1. Set up a virtual environment
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
pip install -r virtualenv/requirements.txt
```

### 2. Download the Spleen Bundle
```
python -m monai.bundle download --name "spleen_ct_segmentation_v0.1.0" --bundle_dir ./job/app/config
``` 

### 3. Download the data
Download the spleen CT data from the [MSD challenge](http://medicaldecathlon.com/) and update data path.

**Note:** The dataset will be saved under `./data`. 
```
python download_spleen_dataset.py
sed -i "s|/workspace/data/Task09_Spleen|${PWD}/data/Task09_Spleen|g" job/app/config/spleen_ct_segmentation/configs/train.json
```

### 4. Run NVFlare in POC mode
To run FL experiments in POC mode, create your local FL workspace the below command.  
```
nvflare poc -n 2 --prepare
```
Then, start the FL system using
```
nvflare poc --start
```

### 5. Run the experiment
Submit the job by running:
```
./submit_job.sh job
```
To monitor the training job, you can start tensorboard:
```
tensorboard --logdir /tmp/nvflare/poc/
```
With the default setting, the expected TensorBoard training curves look like this:

![training curve](./tb_plot.png)

### 6. Shut down the server/clients

To shut down the clients and server, run the following Admin commands:
```
shutdown client
shutdown server
```

> **_NOTE:_** For more information about the Admin client, see [here](https://nvflare.readthedocs.io/en/main/user_guide/operation.html).

## Required NVFLARE version
pip3 install nvflare>=2.2

## (Optional) Install MONAI-NVFlare integration from source
Install `monai_nvflare`:
```
pip install -e .
```
