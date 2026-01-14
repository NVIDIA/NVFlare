
# Hello Differential Privacy

This example demonstrates how to use NVIDIA FLARE with PyTorch and **Differential Privacy (DP)** to train a regression model using federated averaging (FedAvg) with privacy guarantees. The example uses **Opacus** to implement DP-SGD (Differentially Private Stochastic Gradient Descent) during local client training.

## What is Differential Privacy?

[Differential Privacy (DP)](https://arxiv.org/abs/1607.00133) is a mathematical framework that provides strong privacy guarantees when handling sensitive data. In Federated Learning, DP protects user information by adding carefully calibrated noise to model updates during training.

**DP-SGD** adds noise during each optimization step:
1. **Gradient Clipping**: Gradients are clipped to bound sensitivity
2. **Noise Addition**: Gaussian noise is added to clipped gradients
3. **Privacy Accounting**: Privacy budget (ε, δ) is tracked

The privacy-utility trade-off is controlled by epsilon (ε):
- **Lower ε** = Stronger privacy, more noise, lower accuracy
- **Higher ε** = Weaker privacy, less noise, higher accuracy

## NVIDIA FLARE Installation

For complete installation instructions, see [Installation](https://nvflare.readthedocs.io/en/main/installation.html)

```bash
pip install nvflare
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Code Structure

First get the example code from github:

```bash
git clone https://github.com/NVIDIA/NVFlare.git
```

Then navigate to the hello-dp directory:

```bash
git switch <release branch>
cd examples/hello-world/hello-dp
```

```bash
hello-dp
|
|-- client.py             # client training script with DP-SGD using Opacus
|-- model.py              # MLP model definition for tabular data
|-- job.py                # job recipe that defines client and server configurations
|-- hello-dp.ipynb        # Jupyter notebook version
|-- requirements.txt      # dependencies
```

## Data

This example uses the [California Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) - a regression problem to predict median house values in California districts.

**Dataset characteristics:**
- 20,640 samples
- 8 features (median income, house age, average rooms, etc.)
- 1 target (median house value)

In a real FL experiment, each client would have their own dataset. For this example, the dataset is automatically partitioned across clients, so each client has a non-overlapping subset of the data.

## Model

The model is a simple Multi-Layer Perceptron (MLP) for tabular data regression. The implementation can be found in [model.py](model.py).

```python
class TabularMLP(nn.Module):
    """Simple Multi-Layer Perceptron for tabular data regression"""
    
    def __init__(self, input_dim=8, hidden_dims=[64, 32], output_dim=1):
        super(TabularMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
```

The architecture:
- **Input layer**: 8 features
- **Hidden layers**: 64 → 32 neurons with ReLU activation and dropout
- **Output layer**: 1 neuron (house price prediction)

## Client Code with Differential Privacy

The client code `client.py` implements DP-SGD using **Opacus**. The key difference from standard training is adding the `PrivacyEngine`:

```python
from opacus import PrivacyEngine
import nvflare.client as flare

# Initialize NVFlare client
flare.init()

while flare.is_running():
    # Receive model from server
    input_model = flare.receive()
    model.load_state_dict(input_model.params)
    
    # === Apply Differential Privacy ===
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=1.1,  # Controls noise level
        max_grad_norm=1.0,      # Gradient clipping threshold
    )
    # ==================================
    
    # Train as usual - PrivacyEngine handles gradient clipping & noise
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    
    # Check privacy budget spent
    epsilon = privacy_engine.get_epsilon(delta)
    print(f"Privacy spent: (ε = {epsilon:.2f}, δ = {delta})")
    
    # Send model back (note: use _module to get original model)
    output_model = flare.FLModel(
        params=model._module.state_dict(),
        metrics={"rmse": rmse, "privacy_epsilon": epsilon}
    )
    flare.send(output_model)
```

The `PrivacyEngine.make_private()` method:
1. Wraps the model to enable per-sample gradient computation
2. Modifies the optimizer to clip gradients and add noise
3. Wraps the data loader for privacy accounting

## Server-Side Workflow

This example uses the [`FedAvgRecipe`](https://nvflare.readthedocs.io/en/main/apidocs/nvflare.app_opt.pt.recipes.fedavg.html), which implements the [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a) algorithm. The Recipe API handles all server-side logic automatically:

1. Initialize the global model
2. For each training round:
   - Sample available clients
   - Send the global model to selected clients
   - Wait for client updates
   - Aggregate client models into a new global model

With the Recipe API, **there is no need to write custom server code**. The federated averaging workflow is provided by NVFlare.

## Job Recipe Code

The `FedAvgRecipe` combines the client training script with DP parameters:

```python
recipe = FedAvgRecipe(
    name="hello-dp",
    min_clients=n_clients,
    num_rounds=num_rounds,
    initial_model=TabularMLP(input_dim=8, hidden_dims=[64, 32], output_dim=1),
    train_script="client.py",
    train_args=f"--batch_size {batch_size} --target_epsilon {target_epsilon} --n_clients {n_clients}",
)

env = SimEnv(num_clients=n_clients)
recipe.execute(env=env)
```

## Run Job

From the terminal, run:

```bash
python job.py
```

To customize parameters:

```bash
python job.py --n_clients 2 --num_rounds 5 --target_epsilon 50.0
```

Parameters:
- `--n_clients`: Number of federated clients (default: 2)
- `--num_rounds`: Number of federated rounds (default: 5)
- `--batch_size`: Training batch size (default: 64)
- `--target_epsilon`: Target privacy budget - **lower values = stronger privacy** (default: 50.0)
- `--cross_site_eval`: Enable cross-site evaluation after training

To run with cross-site evaluation:

```bash
python job.py --cross_site_eval
```

The cross-site evaluation results can be viewed with:

```bash
cat /tmp/nvflare/simulation/hello-dp/server/simulate_job/cross_site_val/cross_val_results.json
```

## Visualize Results

View training metrics and privacy budget in TensorBoard:

```bash
tensorboard --logdir /tmp/nvflare/simulation/hello-dp
```

Open http://localhost:6006 to see:
- Training loss over time
- RMSE (Root Mean Squared Error)
- Privacy epsilon spent per client

## Notebook

For an interactive version of this example, see [hello-dp.ipynb](./hello-dp.ipynb), which can be executed in Google Colab.

## Privacy-Utility Trade-off

Differential Privacy involves a trade-off between privacy and model utility:

| Epsilon (ε) | Privacy Level | Model Accuracy |
|-------------|---------------|----------------|
| ε = 10      | Strong        | Lower          |
| ε = 50      | Moderate      | Good (default) |
| ε = 100+    | Weak          | Higher         |

**Recommendations:**
- Start with `--target_epsilon 50.0` for a balanced trade-off
- For sensitive data (medical, financial), use ε < 10
- Adjust `max_grad_norm` (gradient clipping) to control sensitivity
- Pre-train on public data before fine-tuning on private data

## Output Summary

#### Initialization
* **TensorBoard**: Logs available at /tmp/nvflare/simulation/hello-dp/server/simulate_job/tb_events
* **Workflow**: FedAvg controller initialized with DP-enabled clients

#### Each Round
* **Model Distribution**: Global model sent to clients
* **Local Training**: Each client trains with DP-SGD using Opacus
* **Privacy Tracking**: Epsilon (ε) logged for each client
* **Aggregation**: DP-trained models aggregated on server

#### Completion
* **Final Model**: Trained model with privacy guarantees
* **Privacy Budget**: Total privacy spent tracked per client

## References

1. Abadi, M., et al. (2016). [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133). ACM CCS 2016.
2. McMahan, B., et al. (2017). [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a). AISTATS 2017.
3. [Opacus: User-friendly library for training PyTorch models with differential privacy](https://opacus.ai/)
4. [NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/)
