# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Client-side training script with Differential Privacy (DP-SGD) using Opacus
"""

import argparse

import torch
import torch.nn as nn
from model import TabularMLP
from opacus import PrivacyEngine
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter


def load_data(client_id, n_clients, batch_size):
    """Load and partition California Housing dataset"""
    # Load dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Split data across clients
    client_data_size = len(X) // n_clients
    start_idx = client_id * client_data_size
    end_idx = (client_id + 1) * client_data_size if client_id < n_clients - 1 else len(X)
    
    X_client = X[start_idx:end_idx]
    y_client = y[start_idx:end_idx]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_client, y_client, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def evaluate(net, data_loader, device):
    """Evaluate model using Mean Squared Error"""
    net.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
    
    mse = total_loss / len(data_loader.dataset)
    rmse = mse ** 0.5
    print(f"Test RMSE: {rmse:.4f}")
    return rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_epsilon", type=float, default=50.0, 
                        help="Target epsilon for differential privacy (lower = more private)")
    parser.add_argument("--target_delta", type=float, default=1e-5,
                        help="Target delta for differential privacy")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--n_clients", type=int, default=2,
                        help="Total number of clients")
    args = parser.parse_args()
    
    batch_size = args.batch_size
    epochs = args.epochs
    lr = 0.01
    
    # Model definition
    model = TabularMLP(input_dim=8, hidden_dims=[64, 32], output_dim=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    # (2) Initialize NVFlare client API
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]
    
    # Extract client ID from site name (e.g., "site-1" -> 0)
    client_id = int(client_name.split("-")[-1]) - 1
    
    # Load data for this client
    train_loader, test_loader = load_data(client_id, args.n_clients, batch_size)
    
    # (optional) metrics tracking
    summary_writer = SummaryWriter()
    
    print(f"Client {client_name}: Using Differential Privacy with epsilon={args.target_epsilon}")
    
    while flare.is_running():
        # (3) Receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"site = {client_name}, current_round={input_model.current_round}")
        
        # (4) Loads model from NVFlare
        model.load_state_dict(input_model.params)
        model.to(device)
        
        # (5) Evaluate on received model for model selection
        rmse = evaluate(model, test_loader, device)
        
        # (optional) Task branch for cross-site evaluation
        if flare.is_evaluate():
            print(f"site = {client_name}, running cross-site evaluation")
            # For CSE, just return the evaluation metrics without training
            output_model = flare.FLModel(metrics={"rmse": rmse})
            flare.send(output_model)
            continue
        
        # ====== DIFFERENTIAL PRIVACY SETUP ======
        # Attach PrivacyEngine to the model, optimizer, and data loader
        privacy_engine = PrivacyEngine()
        
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=1.1,
            max_grad_norm=args.max_grad_norm,
        )
        # ========================================
        
        model.train()
        steps = epochs * len(train_loader)
        
        for epoch in range(epochs):
            running_loss = 0.0
            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if i % 10 == 9:  # Print every 10 batches
                    avg_loss = running_loss / 10
                    
                    # Optional: Log metrics
                    global_step = input_model.current_round * steps + epoch * len(train_loader) + i
                    summary_writer.add_scalar(tag="train_loss", scalar=avg_loss, global_step=global_step)
                    
                    print(f"site={client_name}, Epoch: {epoch + 1}/{epochs}, Batch: {i + 1}, Loss: {avg_loss:.4f}")
                    running_loss = 0.0
        
        # Print privacy budget spent
        epsilon = privacy_engine.get_epsilon(args.target_delta)
        print(f"Client {client_name}: Privacy spent - (ε = {epsilon:.2f}, δ = {args.target_delta})")
        summary_writer.add_scalar(tag="privacy_epsilon", scalar=epsilon, 
                                   global_step=input_model.current_round)
        
        print(f"Finished Training for {client_name}")
        
        # (6) Construct trained FL model
        output_model = flare.FLModel(
            params=model._module.cpu().state_dict(),  # Note: use _module to get original model from PrivacyEngine
            metrics={"rmse": rmse, "privacy_epsilon": epsilon},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        print(f"site: {client_name}, sending model to server.")
        
        # (7) Send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
