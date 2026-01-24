#!/usr/bin/env python3
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

"""Memory profiling test for FedAvg with/without server_memory_gc_rounds.

This script compares memory usage (RSS) with different cleanup settings:
- server_memory_gc_rounds=0 (disabled)
- server_memory_gc_rounds=5 (every 5 rounds)
- server_memory_gc_rounds=1 (every round)

Usage:
    python test_fedavg_memory.py

    # With memory_profiler for detailed analysis:
    mprof run test_fedavg_memory.py
    mprof plot
"""

import gc
import os
import sys
import tempfile
from pathlib import Path

import psutil


def get_rss_mb() -> float:
    """Get current process RSS in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def create_test_model(size_mb: int = 50) -> dict:
    """Create a test model of approximately the specified size in MB.
    
    Args:
        size_mb: Approximate size of the model in MB.
        
    Returns:
        Dictionary representing model parameters.
    """
    # Each float is 8 bytes, so for size_mb MB we need (size_mb * 1024 * 1024) / 8 floats
    floats_per_layer = 100000  # ~0.8 MB per layer
    num_layers = max(1, (size_mb * 1024 * 1024) // (floats_per_layer * 8))
    
    return {f"layer_{i}": [0.0] * floats_per_layer for i in range(num_layers)}


def create_client_script(tmpdir: Path) -> Path:
    """Create a minimal client training script.
    
    Args:
        tmpdir: Temporary directory to store the script.
        
    Returns:
        Path to the created script.
    """
    client_script = tmpdir / "client.py"
    client_script.write_text('''
import nvflare.client as flare

flare.init()

while flare.is_running():
    input_model = flare.receive()
    
    # Simulate some training (just pass through for memory test)
    output_model = flare.FLModel(
        params=input_model.params,
        params_type=input_model.params_type,
    )
    
    flare.send(output_model)
''')
    return client_script


def run_simulation(
    server_memory_gc_rounds: int,
    num_rounds: int = 10,
    num_clients: int = 2,
    model_size_mb: int = 50,
) -> dict:
    """Run FedAvg simulation with specified memory settings.
    
    Args:
        server_memory_gc_rounds: Cleanup frequency (0=disabled, N=every N rounds).
        num_rounds: Number of FL rounds to run.
        num_clients: Number of simulated clients.
        model_size_mb: Approximate model size in MB.
        
    Returns:
        Dictionary with memory statistics.
    """
    from nvflare.recipe.fedavg import FedAvgRecipe
    from nvflare.recipe.sim_env import SimEnv

    print(f"\n{'=' * 60}")
    print(f"Testing: server_memory_gc_rounds={server_memory_gc_rounds}")
    print(f"Rounds: {num_rounds}, Clients: {num_clients}, Model: ~{model_size_mb}MB")
    print(f"{'=' * 60}")

    # Force GC before starting
    gc.collect()
    initial_rss = get_rss_mb()
    print(f"Initial RSS: {initial_rss:.1f} MB")

    # Create test model
    initial_model = create_test_model(model_size_mb)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        client_script = create_client_script(tmpdir_path)

        recipe = FedAvgRecipe(
            name=f"memory_test_gc_{server_memory_gc_rounds}",
            min_clients=num_clients,
            num_rounds=num_rounds,
            train_script=str(client_script),
            initial_model=initial_model,
            server_memory_gc_rounds=server_memory_gc_rounds,
        )

        env = SimEnv(num_clients=num_clients)
        
        # Run the simulation
        run = recipe.execute(env)

    # Force GC after run
    gc.collect()
    final_rss = get_rss_mb()
    rss_increase = final_rss - initial_rss

    print(f"Final RSS: {final_rss:.1f} MB")
    print(f"RSS increase: {rss_increase:.1f} MB")

    return {
        "server_memory_gc_rounds": server_memory_gc_rounds,
        "initial_rss_mb": initial_rss,
        "final_rss_mb": final_rss,
        "rss_increase_mb": rss_increase,
    }


def print_summary(results: list):
    """Print summary comparison of all test runs.
    
    Args:
        results: List of result dictionaries from run_simulation.
    """
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Setting':<30} {'Initial MB':>12} {'Final MB':>12} {'Increase':>12}")
    print("-" * 66)
    
    for r in results:
        gc_setting = r["server_memory_gc_rounds"]
        label = f"gc_rounds={gc_setting}" if gc_setting > 0 else "gc_rounds=0 (disabled)"
        print(
            f"{label:<30} "
            f"{r['initial_rss_mb']:>12.1f} "
            f"{r['final_rss_mb']:>12.1f} "
            f"{r['rss_increase_mb']:>+12.1f}"
        )
    
    print("-" * 66)
    
    # Calculate improvement
    if len(results) >= 2:
        baseline = results[0]["rss_increase_mb"]
        for r in results[1:]:
            if baseline > 0:
                improvement = (baseline - r["rss_increase_mb"]) / baseline * 100
                print(f"gc_rounds={r['server_memory_gc_rounds']} vs disabled: {improvement:.1f}% reduction")


def main():
    """Run memory profiling tests."""
    print("FedAvg Memory Profiling Test")
    print(f"MALLOC_ARENA_MAX: {os.environ.get('MALLOC_ARENA_MAX', 'not set')}")
    print(f"Python: {sys.version}")
    
    # Configuration
    num_rounds = 10
    num_clients = 2
    model_size_mb = 50
    
    results = []
    
    # Test 1: No cleanup (disabled)
    results.append(run_simulation(
        server_memory_gc_rounds=0,
        num_rounds=num_rounds,
        num_clients=num_clients,
        model_size_mb=model_size_mb,
    ))
    
    # Test 2: Cleanup every 5 rounds (recommended for server)
    results.append(run_simulation(
        server_memory_gc_rounds=5,
        num_rounds=num_rounds,
        num_clients=num_clients,
        model_size_mb=model_size_mb,
    ))
    
    # Test 3: Cleanup every round
    results.append(run_simulation(
        server_memory_gc_rounds=1,
        num_rounds=num_rounds,
        num_clients=num_clients,
        model_size_mb=model_size_mb,
    ))
    
    print_summary(results)


if __name__ == "__main__":
    main()

