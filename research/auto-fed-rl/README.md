# Auto-FedRL

Components for running the algorithm proposed in 
["Auto-FedRL: Federated Hyperparameter Optimization for Multi-institutional Medical Image Segmentation"](https://arxiv.org/abs/2203.06338) (ECCV 2022)
with [NVIDIA FLARE](https://nvflare.readthedocs.io/en/main/index.html).

You can find the original experimental implementation [here](https://github.com/guopengf/Auto-FedRL).

> **Note:** For more examples of FL algorithms with the CIFAR-10 dataset, 
> see [here](../../examples/cifar10).

## (Optional) 1. Set up a virtual environment
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
pip install -r ./virtualenv/requirements.txt
```
Set `PYTHONPATH` to include custom files of this example:

We use both the utils from the [CIFAR-10 examples](../../examples/cifar10) 
and files under [./src](./src):
```
export PYTHONPATH=${PWD}/../../examples/cifar10:${PWD}/../../examples/cifar10/pt/utils:${PWD}/src
```

### 2. Download the CIFAR-10 dataset 
To speed up the following experiments, first download the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset:
```
python3 -m pt.utils.cifar10_download_data
```

## 3. Run simulated FL experiments

We are using NVFlare's [FL simulator](https://nvflare.readthedocs.io/en/latest/user_guide/fl_simulator.html) to run the following experiments. 

The output root of where to save the results is set in [./run_simulator.sh](./run_simulator.sh) as `RESULT_ROOT=/tmp/nvflare/sim_cifar10`.

### 3.1 Varying data heterogeneity of data splits

We use an implementation to generated heterogeneous data splits from CIFAR-10 based a Dirichlet sampling strategy 
from FedMA (https://github.com/IBM/FedMA), where `alpha` controls the amount of heterogeneity, 
see [Wang et al.](https://arxiv.org/abs/2002.06440).

We use `set_alpha.sh` to change the alpha value inside the job configurations.

### 3.2 Auto-FedRL

Next, let's try to run Auto-FedRL on a heterogeneous split (alpha=0.1):

[Auto-FedRL](https://arxiv.org/abs/2203.06338) is a method for federated hyperparameter optimization
```
./run_simulator.sh cifar10_autofedrl 0.1 8 8
```
