## 3D spleen CT segmentation
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
python -m monai.bundle download --name "spleen_ct_segmentation_v0.1.1" --bundle_dir ./job/app/config
``` 

### 3. Download the data
Download the spleen CT data from the [MSD challenge](http://medicaldecathlon.com/) and update data path.

> **Note:** The dataset will be saved under `./data`. 
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
nvflare poc --stop
```

> **_NOTE:_** For more information about the Admin client, see [here](https://nvflare.readthedocs.io/en/main/user_guide/operation.html).
