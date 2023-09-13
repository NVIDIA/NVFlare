# TensorFlow Deep Learning to Federated Learning transition with NVFlare

Please install the requirements first, it is suggested to install inside a virtual environment:

```bash
pip install -r requirements.txt
```

## Transform CIFAR10 Tensorflow training code to FL with NVFLARE Client API

Given a Tensorflow CIFAR10 example: [./code/cifar10_tf_original.py](./code/cifar10_tf_original.py).

You can run it using

```bash
python3 ./code/cifar10_tf_original.py
```

To transform the existing code to FL training code, we made the following changes:

1. Import NVFlare Client API: ```import nvflare.client as flare```
2. Initialize NVFlare Client API: ```flare.init()```
3. Receive aggregated/global FLModel from NVFlare side: ```input_model = flare.receive()```
4. Load the received aggregated/global model weights into the model structure: ```load_flat_weights(net, input_model.params)```
5. Evaluate on received aggregated/global model to get the metrics for model selection
6. Construct the FLModel to be returned to the NVFlare side: ```output_model = flare.FLModel(params=get_flat_weights(net), xxx)```
7. Send the model back to NVFlare: ```flare.send(output_model)```

Notice that we need to flatten/unflatten the model weights because NVFlare server-side aggregators now
only accept a ``dict`` of arrays.

The modified code can be found here: [./code/cifar10_tf_fl.py](./code/cifar10_tf_fl.py), [./code/tf_net.py](./code/tf_net.py).

After we modify our training script, we need to put it into a [job structure](https://nvflare.readthedocs.io/en/latest/real_world_fl/job.html) so that NVFlare system knows how to deploy and run the job.

Please refer to [JOB CLI tutorial](../../../tutorials/job_cli.ipynb) on how to generate a job easily from our existing job templates.


We choose the [tensorflow job template](../../../../job_templates/sag_tf/) and run the following command to create the job:

```bash
nvflare config -jt ../../../../job_templates
nvflare job create -force -j ./jobs/tensorflow -w sag_tf -sd ./code/ -s ./code/cifar10_tf_fl.py
```

Then we can run the job using the simulator:

```bash
bash ../prepare_data.sh
nvflare simulator -n 2 -t 2 ./jobs/tensorflow -w tensorflow_workspace
```
