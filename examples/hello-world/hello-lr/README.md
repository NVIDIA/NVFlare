# Federated Logistic Regression with Second-Order Newton-Raphson optimization

This example shows how to implement a federated binary
classification via logistic regression with second-order Newton-Raphson optimization.

## Install NVFLARE and Dependencies

for the complete installation instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html)
```
pip install nvflare

```
Get the example code from github:

```
  git clone https://github.com/NVIDIA/NVFlare.git
```

then navigate to the hello-lr directory:

```
  git switch <release branch>
  cd examples/hello-world/hello-lr
```

Install the dependency

```
  pip install -r requirements.txt
```

## Code Structure

``` bash
hello-lr
    |
    |-- client.py         # client local training script
    |-- job.py            # job recipe that defines client and server configurations
    |-- download_data.py  # download dataset
    |-- prepare_data.py   # prepare data to convert to numpy
    |-- requirements.txt  # dependencies
```


## Data

The [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) is
used in this example. 


**Publication Request:**

This file describes the contents of the heart-disease directory.
 
The authors of the databases have requested:

      ...that any publications resulting from the use of the data include the 
      names of the principal investigator responsible for the data collection
      at each institution.  They would be:

       1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
       2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
       3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
       4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:
	  Robert Detrano, M.D., Ph.D.
 

dataset contains samples from 4 sites, split into training and
testing sets as described below:

|site         | sample split                          |
|-------------|---------------------------------------|
|Cleveland    | train: 199 samples, test: 104 samples |
|Hungary      | train: 172 samples, test: 89 samples  |
|Switzerland  | train: 30 samples, test: 16 samples   |
|Long Beach V | train: 85 samples, test: 45 samples   |

The number of features in each sample is 13.


### Features 

| Variable Name | Role    | Type        | Demographic | Description                                           | Units  | Missing Values |
|---------------|---------|-------------|-------------|-------------------------------------------------------|--------|----------------|
| age           | Feature | Integer     | Age         | years                                                 |        | no             |
| sex           | Feature | Categorical | Sex         |                                                       |        | no             |
| cp            | Feature | Categorical |             |                                                       |        | no             |
| trestbps      | Feature | Integer     |             | resting blood pressure (on admission to the hospital) | mm Hg  | no             |
| chol          | Feature | Integer     |             | serum cholestoral                                     | mg/dl  | no             |
| fbs           | Feature | Categorical |             | fasting blood sugar > 120 mg/dl                       |        | no             |
| restecg       | Feature | Categorical |             |                                                       |        | no             |
| thalach       | Feature | Integer     |             | maximum heart rate achieved                           |        | no             |
| exang         | Feature | Categorical |             | exercise induced angina                               |        | no             |
| oldpeak       | Feature | Integer     |             | ST depression induced by exercise relative to rest    |        | no             |
| slope         | Feature | Categorical |             |                                                       |        | no             |
| ca            | Feature | Integer     |             | number of major vessels (0-3) colored by flourosopy   |        | yes            |
| thal          | Feature | Categorical |             |                                                       |        | yes            |
| num           | Target  | Integer     |             | diagnosis of heart disease                            |        | no             |

## Model

The [Newton-Raphson optimization](https://en.wikipedia.org/wiki/Newton%27s_method) problem
can be described as follows.

In a binary classification task with logistic regression, the
probability of a data sample $x$ classified as positive is formulated
as:
$$p(x) = \sigma(\beta \cdot x + \beta_{0})$$
where $\sigma(.)$ denotes the sigmoid function. We can incorporate
$\beta_{0}$ and $\beta$ into a single parameter vector $\theta =
( \beta_{0},  \beta)$. Let $d$ be the number
of features for each data sample $x$ and let $N$ be the number of data
samples. We then have the matrix version of the above probability
equation:
$$p(X) = \sigma( X \theta )$$
Here $X$ is the matrix of all samples, with shape $N \times (d+1)$,
having it's first column filled with value 1 to account for the
intercept $\theta_{0}$.

The goal is to compute parameter vector $\theta$ that maximizes the
below likelihood function:
$$L_{\theta} = \prod_{i=1}^{N} p(x_i)^{y_i} (1 - p(x_i)^{1-y_i})$$

The Newton-Raphson method optimizes the likelihood function via
quadratic approximation. Omitting the maths, the theoretical update
formula for parameter vector $\theta$ is:
$$\theta^{n+1} = \theta^{n} - H_{\theta^{n}}^{-1} \nabla L_{\theta^{n}}$$
where
$$\nabla L_{\theta^{n}} = X^{T}(y - p(X))$$
is the gradient of the likelihood function, with $y$ being the vector
of ground truth for sample data matrix $X$,  and
$$H_{\theta^{n}} = -X^{T} D X$$
is the Hessian of the likelihood function, with $D$ a diagonal matrix
where diagonal value at $(i,i)$ is $D(i,i) = p(x_i) (1 - p(x_i))$.

In federated Newton-Raphson optimization, each client will compute its
own gradient $\nabla L_{\theta^{n}}$ and Hessian $H_{\theta^{n}}$
based on local training samples. A server will aggregate the gradients
and Hessians computed from all clients, and perform the update of
parameter $\theta$ based on the theoretical update formula described
above.

## Client Side

On the client side, the local training logic is implemented
[client.py](./client.py). The implementation is based on the [`Client
API`](https://nvflare.readthedocs.io/en/main/programming_guide/execution_api_type.html#client-api). This
allows user to add minimum `nvflare`-specific codes to turn a typical
centralized training script to a federated client side local training
script.
- During local training, each client receives a copy of the global
  model, sent by the server, using `flare.receive()` API. The received
  global model is an instance of `FLModel`.
- A local validation is first performed, where validation metrics
  
- Then each client computes it's gradient and Hessian based on local
  training data, using their respective theoretical formula described
  above. This is implemented in the
  [`train_newton_raphson()`](./client.py) method. Each client then 
  sends the computed results (always in `FLModel` format) to server for aggregation, 
  using `flare.send()` API.

Each client site corresponds to a site listed in the data table above.

The training logic remains similar to the centralized logic: load data, perform training
(Newton-Raphson updates), and valid trained model. The only added
differences in the federated code are related to interaction with the
FL system, such as receiving and send `FLModel`.


## Server Side
We leverage a builtin FLARE logistic regression with Newton Raphson method. 
the server side fedavg class is located at `nvflare.app_common.workflows.lr.fedavg.FedAvgLR`

## Job
```
  recipe = FedAvgLrRecipe(
  num_rounds=num_rounds,
  damping_factor=0.8,
  num_features=13,
  train_script="client.py",
  train_args=f"--data_root {data_root}",
  )
  env = SimEnv(num_clients=n_clients, num_threads=n_clients)
  run = recipe.execute(env)
  # run.get_result()
```

## Download and prepare data

Execute the following script
```
python download_data.py 
python prepare_data.py 
```
This will download the heart disease dataset under
`/tmp/flare/dataset/heart_disease_data/`


## Running Job 

Execute the following command to launch federated logistic
regression. This will run in nvflare's simulation mode.
```
python job.py
```
