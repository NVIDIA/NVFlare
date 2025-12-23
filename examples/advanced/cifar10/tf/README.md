# Getting Started with NVFlare (TensorFlow)
[![TensorFlow Logo](https://upload.wikimedia.org/wikipedia/commons/a/ab/TensorFlow_logo.svg)](https://tensorflow.org/)

We provide several examples to help you quickly get started with NVFlare.
All examples in this folder are based on using [TensorFlow](https://tensorflow.org/) as the model training framework.

## Simulated Federated Learning with CIFAR10 Using TensorFlow

This example demonstrates TensorFlow-based federated learning algorithms,
including FedAvg, FedOpt, FedProx, and SCAFFOLD, on the CIFAR-10 dataset.

The example is structured with separate folders for each algorithm, similar to the PyTorch examples:
- **cifar10_central**: Centralized training baseline
- **cifar10_fedavg**: FedAvg algorithm implementation
- **cifar10_fedopt**: FedOpt algorithm implementation
- **cifar10_fedprox**: FedProx algorithm implementation
- **cifar10_scaffold**: SCAFFOLD algorithm implementation

Each algorithm folder contains:
- `client.py`: Client-side training logic
- `job.py`: Job configuration and execution script
- `README.md`: Algorithm-specific documentation


## 1. Install requirements

Install required packages:
```bash
pip install --upgrade pip
pip install -r ./requirements.txt
```

> **_NOTE:_**  We recommend either using a containerized deployment or virtual environment,
> please refer to [getting started](https://nvflare.readthedocs.io/en/latest/getting_started.html).

## 2. Run experiments

This example uses simulator to run all experiments. Each algorithm folder contains
a `job.py` script that can be run independently. A master script
`run_experiments.sh` is also provided to run all experiments at once:

```bash
bash ./run_experiments.sh
```

The CIFAR10 dataset will be downloaded when running any experiment for
the first time. `TensorBoard` summary logs will be generated during
any experiment, and you can use `TensorBoard` to visualize the
training and validation process as the experiment runs. Data split
files, summary logs and results will be saved in a workspace
directory, which defaults to `/tmp` and can be configured by setting
the `--workspace` argument in each `job.py` script.

> [!WARNING]
> If you are using GPU, make sure to set the following
> environment variables before running a training job, to prevent
> `TensorFlow` from allocating full GPU memory all at once:
> `export TF_FORCE_GPU_ALLOW_GROWTH=true && export TF_GPU_ALLOCATOR=cuda_malloc_asyncp`

We apply Dirichlet sampling (as implemented in FedMA: https://github.com/IBM/FedMA) to
CIFAR10 data labels to simulate data heterogeneity among client sites, controlled by an
alpha value between 0 (exclusive) and 1. A high alpha value indicates less data
heterogeneity, i.e., an alpha value equal to 1.0 would result in homogeneous data 
distribution among different splits.

### 2.1 Centralized training

To simulate a centralized training baseline, we train on the full dataset for 25 epochs.

```bash
python cifar10_central/train.py --epochs 25
```

### 2.2 FedAvg with different data heterogeneity (alpha values)

Here we run FedAvg for 50 rounds, each round with 4 local epochs. This
corresponds roughly to the same number of iterations across clients as
in the centralized baseline above (50*4 divided by 8 clients is 25):

```bash
python cifar10_fedavg/job.py --n_clients 8 --num_rounds 50 --alpha 1.0
python cifar10_fedavg/job.py --n_clients 8 --num_rounds 50 --alpha 0.5
python cifar10_fedavg/job.py --n_clients 8 --num_rounds 50 --alpha 0.3
python cifar10_fedavg/job.py --n_clients 8 --num_rounds 50 --alpha 0.1
```

### 2.3 FedProx

FedProx adds a proximal term to the local objective function to handle data heterogeneity:

```bash
python cifar10_fedprox/job.py --n_clients 8 --num_rounds 50 --alpha 0.1 --fedprox_mu 0.1
```

### 2.4 FedOpt

FedOpt uses adaptive optimizers (e.g., SGD with momentum) on the server side:

```bash
python cifar10_fedopt/job.py --n_clients 8 --num_rounds 50 --alpha 0.1
```

### 2.5 SCAFFOLD

SCAFFOLD uses control variates to correct for client drift:

```bash
python cifar10_scaffold/job.py --n_clients 8 --num_rounds 50 --alpha 0.1
```

## 3. Results

Now let's compare experimental results.

### 3.1 Centralized training vs. FedAvg for homogeneous split

Let's first compare FedAvg with homogeneous data split
(i.e. `alpha=1.0`) and centralized training. As can be seen from the
figure and table below, FedAvg can achieve similar performance to
centralized training under homogeneous data split, i.e., when there is
no difference in data distributions among different clients.

| Config          | Alpha | Val score |
|-----------------|-------|-----------|
| cifar10_central | n.a.  | 0.8758    |
| cifar10_fedavg  | 1.0   | 0.8839    |

![Central vs. FedAvg](./figs/fedavg-vs-centralized.png)

### 3.2 Impact of client data heterogeneity

Here we compare the impact of data heterogeneity by varying the
`alpha` value, where lower values cause higher heterogeneity. As can
be observed in the table below, performance of the FedAvg decreases
as data heterogeneity becomes higher.

| Config         | Alpha | Val score |
|----------------|-------|-----------|
| cifar10_fedavg | 1.0   | 0.8838    |
| cifar10_fedavg | 0.5   | 0.8685    |
| cifar10_fedavg | 0.3   | 0.8323    |
| cifar10_fedavg | 0.1   | 0.7903    |

![Impact of client data heterogeneity](./figs/fedavg-diff-alphas.png)

### 3.3 Impact of different FL algorithms

Lastly, we compare the performance of different FL algorithms, with
`alpha` value fixed to 0.1, i.e., a high client data heterogeneity.
We can observe from the figure below that, FedOpt and
SCAFFOLD achieve better performance, with better convergence rates
compared to FedAvg and FedProx with the same alpha setting. SCAFFOLD
achieves that by adding a correction term when updating the client
models, while FedOpt utilizes SGD with momentum to update the global
model on the server. Both achieve better performance with the same
number of training steps as FedAvg/FedProx.

| Config            | Alpha | Val score |
|-------------------|-------|-----------|
| cifar10_fedavg    | 0.1   | 0.7903    |
| cifar10_fedopt    | 0.1   | 0.8145    |
| cifar10_fedprox   | 0.1   | 0.7843    |
| cifar10_scaffold  | 0.1   | 0.8164    |

![Impact of different FL algorithms](./figs/fedavg-diff-algos.png)

> [!NOTE]
> More examples can be found at https://nvidia.github.io/NVFlare.
