# NVIDIA FLARE Hello World Examples

Welcome to the NVIDIA FLARE Hello World examples! These examples demonstrate how to use NVIDIA FLARE's **Job Recipe** API to quickly build and run federated learning applications across different frameworks.

## Quick Start

### Prerequisites

1. **Install NVIDIA FLARE:**
   ```bash
   pip install nvflare
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/NVIDIA/NVFlare.git
   cd NVFlare/examples/hello-world
   ```

3. **Choose an example and install dependencies:**
   ```bash
   cd hello-pt  # or hello-numpy, hello-lightning, hello-tf, hello-flower
   pip install -r requirements.txt
   ```

### Run Your First Example

Running a federated learning job with NVIDIA FLARE is as simple as executing a Python script:

```bash
python job.py
```

That's it! The job will run in simulation mode with multiple clients, and you'll see the training progress in your terminal.

### What Just Happened?

When you ran `python job.py`, NVIDIA FLARE:
1. Created a federated learning job using a **Job Recipe**
2. Started a simulation environment with multiple clients
3. Distributed the model to clients for local training
4. Aggregated the results using FedAvg
5. Saved the results to `/tmp/nvflare/`

## What is a Job Recipe?

A **Job Recipe** is NVIDIA FLARE's high-level API for defining federated learning jobs. Instead of manually configuring workflows, controllers, and executors, you simply specify:

- The model to train
- The training script
- The number of clients and rounds
- The aggregation algorithm (e.g., FedAvg)

Here's a complete example:

```
   from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
   from nvflare.recipe import SimEnv
   
   recipe = FedAvgRecipe(
       name="hello-pt",
       min_clients=2,
       num_rounds=2,
       initial_model=SimpleNetwork(),
       train_script="client.py",
   )
   
   env = SimEnv(num_clients=2)
   run = recipe.execute(env)
```

### Benefits of Job Recipes

- **Simple**: Define complete FL jobs in 5-10 lines of Python code
- **Flexible**: Same recipe works in simulation, POC, and production environments
- **Framework-specific**: Optimized recipes for PyTorch, TensorFlow, NumPy, and more
- **No configuration files**: Everything defined in Python
- **Easy experimentation**: Change parameters and re-run instantly

## Examples by Framework

### Deep Learning Frameworks

#### [Hello PyTorch](./hello-pt/)
Train an image classifier on CIFAR-10 using PyTorch and FedAvg.

**What you'll learn:**
- Using the PyTorch FedAvg Recipe
- Client API for PyTorch models
- TensorBoard integration

**Run it:**
```bash
cd hello-pt
pip install -r requirements.txt
python job.py
```

#### [Hello PyTorch Lightning](./hello-lightning/)
Train an image classifier using PyTorch Lightning with federated learning.

**What you'll learn:**
- Integrating PyTorch Lightning with NVIDIA FLARE
- Using LightningModule and LightningDataModule
- Minimal code changes for FL

**Run it:**
```bash
cd hello-lightning
pip install -r requirements.txt
./prepare_data.sh  # Pre-download CIFAR-10
python job.py
```

#### [Hello TensorFlow](./hello-tf/)
Train an MNIST classifier using TensorFlow and FedAvg.

**What you'll learn:**
- Using the TensorFlow FedAvg Recipe
- Client API for TensorFlow models
- GPU memory management for multi-client scenarios

**Run it:**
```bash
cd hello-tf
pip install -r requirements.txt
TF_FORCE_GPU_ALLOW_GROWTH=true python job.py
```

### Traditional ML

#### [Hello NumPy](./hello-numpy/)
Demonstrate federated averaging with a simple NumPy model.

**What you'll learn:**
- Using the NumPy FedAvg Recipe
- Federated learning without deep learning frameworks
- Basic FL concepts with minimal code

**Run it:**
```bash
cd hello-numpy
pip install -r requirements.txt
python job.py
```

### Framework Integration

#### [Hello Flower](./hello-flower/)
Run Flower applications on NVIDIA FLARE infrastructure.

**What you'll learn:**
- Integrating Flower with NVIDIA FLARE
- Running Flower ClientApp and ServerApp
- Metric streaming with TensorBoard

**Run it:**
```bash
cd hello-flower
pip install -r requirements.txt
python job.py --job_name "flwr-pt" --content_dir "./flwr-pt"
```

## Understanding the Code Structure

Each example follows a consistent structure:

```
hello-<framework>/
├── client.py         # Client-side training code
├── model.py          # Model definition
├── job.py            # Job recipe that creates and runs the FL job
└── requirements.txt  # Dependencies
```

### Client Code (`client.py`)

The client code contains your training logic with minimal NVIDIA FLARE integration:

```
   import nvflare.client as flare
   
   flare.init()  # Initialize FLARE Client API
   
   while flare.is_running():
       input_model = flare.receive()  # Receive global model
       params = input_model.params
       
       # Your training code here
       new_params = train(params)
       
       output_model = flare.FLModel(params=new_params)
       flare.send(output_model)  # Send updated model
```

### Job Recipe (`job.py`)

The job recipe defines the FL workflow:

```
   from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
   from nvflare.recipe import SimEnv
   
   recipe = FedAvgRecipe(
       name="my-job",
       min_clients=2,
       num_rounds=3,
       initial_model=MyModel(),
       train_script="client.py",
   )
   
   env = SimEnv(num_clients=2)
   run = recipe.execute(env)
```

## Additional Examples

### Step-by-Step Examples
Detailed tutorials covering specific FL techniques and workflows:
- [CIFAR-10 Examples](./step-by-step/cifar10/) - FedAvg, Cyclic, Cross-Site Validation, Swarm Learning
- [Higgs Examples](./step-by-step/higgs/) - Scikit-learn, XGBoost, Federated Statistics

[Learn more →](./step-by-step/)

### ML-to-FL Conversion
Learn how to convert existing ML/DL code to federated learning:
- [PyTorch Conversion](./ml-to-fl/pt/)
- [TensorFlow Conversion](./ml-to-fl/tf/)
- [NumPy Conversion](./ml-to-fl/np/)

[Learn more →](./ml-to-fl/)

### Workflows
Examples demonstrating different FL workflows:
- [Scatter and Gather](./hello-numpy/) - Basic FedAvg pattern
- [Cross-Site Validation](./hello-numpy-cross-val/) - Model evaluation across sites
- [Cyclic Weight Transfer](./hello-cyclic/) - Sequential client training
- [Client Controlled Workflows](../advanced/hello-ccwf/) - Swarm learning patterns

## Running with Different Environments

The same Job Recipe can run in different environments by changing the `env` parameter:

### Simulation (Default)
```
   env = SimEnv(num_clients=2)
   recipe.execute(env)
```

### POC Mode
```
   env = PoCEnv()
   recipe.execute(env)
```

### Production
```
   env = ProdEnv()
   recipe.execute(env)
```

## Accessing Results

After running a job, results are saved in the simulator workspace:

```bash
# Default location
ls /tmp/nvflare/simulation/<job-name>/

# Server logs and models
ls /tmp/nvflare/simulation/<job-name>/server/simulate_job/

# TensorBoard logs (if enabled)
tensorboard --logdir /tmp/nvflare/simulation/<job-name>/server/simulate_job/tb_events
```

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Make sure you're in the example directory and have installed requirements
cd hello-pt
pip install -r requirements.txt
```

**Port already in use:**
```bash
# The simulator uses random ports, but if you see port conflicts:
# Kill any running FLARE processes
ps aux | grep nvflare
kill <process_id>
```

**GPU out of memory (TensorFlow):**
```bash
# Use memory growth flags
TF_FORCE_GPU_ALLOW_GROWTH=true python job.py
```

**Data download issues (Lightning/PyTorch):**
```bash
# Pre-download data before running
./prepare_data.sh
```

## Advanced Topics

### FLARE API
For more control over job submission and monitoring, you can use the [FLARE API](../tutorials/flare_api.ipynb). This allows you to:
- Submit jobs programmatically
- Monitor job status
- Retrieve results
- Manage multiple jobs

### POC Mode
For testing in a more production-like environment, you can use [POC mode](../tutorials/setup_poc.ipynb). This sets up a local FL system with separate server and client processes.

### Production Deployment
When you're ready to deploy to production, see the [deployment guide](https://nvflare.readthedocs.io/en/main/user_guide/admin_guide/deployment/overview.html).

## Interactive Notebook

For an interactive walkthrough of these examples, see the [Hello World Notebook](./hello_world.ipynb), which provides step-by-step guidance through each framework.

## Next Steps

1. **Try the examples** - Run each example to understand different frameworks
2. **Modify the code** - Experiment with different models, datasets, and parameters
3. **Explore advanced examples** - Check out [advanced examples](../advanced/) for more complex scenarios
4. **Read the documentation** - Visit [NVFLARE Documentation](https://nvflare.readthedocs.io/) for comprehensive guides

## Getting Help

- **Documentation**: https://nvflare.readthedocs.io/
- **GitHub Issues**: https://github.com/NVIDIA/NVFlare/issues
- **Discussions**: https://github.com/NVIDIA/NVFlare/discussions

## Additional Resources

- [Programming Guide](https://nvflare.readthedocs.io/en/main/programming_guide.html)
- [User Guide](https://nvflare.readthedocs.io/en/main/user_guide.html)
- [API Documentation](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.html)
- [Example Catalog](https://nvidia.github.io/NVFlare/catalog/)