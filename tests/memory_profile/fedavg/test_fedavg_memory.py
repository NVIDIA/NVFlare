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

Each test runs in a separate subprocess for complete isolation.

Usage:
    python test_fedavg_memory.py

    # With memory_profiler for detailed analysis:
    MALLOC_ARENA_MAX=4 mprof run test_fedavg_memory.py
    mprof plot -o memory_profile.png
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import psutil


def get_rss_mb() -> float:
    """Get current process RSS in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def get_peak_rss_mb() -> float:
    """Get peak RSS (max resident set size) in MB.

    Uses resource.getrusage() which tracks the max RSS since process start.
    In subprocess mode, this gives the peak for that specific test.
    """
    import resource

    # ru_maxrss is in KB on Linux, bytes on macOS
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return ru_maxrss / (1024 * 1024)  # bytes to MB
    else:
        return ru_maxrss / 1024  # KB to MB


def create_test_model(size_mb: int = 100) -> dict:
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
    client_script.write_text(
        """
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
"""
    )
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
    import time

    print(f"\n{'=' * 60}")
    print(f"Testing: server_memory_gc_rounds={server_memory_gc_rounds}")
    print(f"Rounds: {num_rounds}, Clients: {num_clients}, Model: ~{model_size_mb}MB")
    print(f"{'=' * 60}")

    # Force GC before starting
    gc.collect()
    initial_rss = get_rss_mb()
    print(f"Initial RSS: {initial_rss:.1f} MB")

    # Stage 1: Import NVFlare modules
    t0 = time.time()
    from nvflare.recipe.fedavg import FedAvgRecipe
    from nvflare.recipe.sim_env import SimEnv

    t1 = time.time()
    print(f"[Timing] Import NVFlare modules: {t1 - t0:.2f}s")

    # Stage 2: Create test model
    initial_model = create_test_model(model_size_mb)
    t2 = time.time()
    print(f"[Timing] Create test model ({model_size_mb}MB): {t2 - t1:.2f}s")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        client_script = create_client_script(tmpdir_path)

        # Stage 3: Create FedAvgRecipe
        recipe = FedAvgRecipe(
            name=f"memory_test_gc_{server_memory_gc_rounds}",
            min_clients=num_clients,
            num_rounds=num_rounds,
            train_script=str(client_script),
            initial_model=initial_model,
            server_memory_gc_rounds=server_memory_gc_rounds,
        )

        # Calculate timeout based on model size and clients
        # streaming_per_request_timeout: default 600s, add buffer for large payloads
        base_timeout = 600  # NVFlare default
        transfer_time = (model_size_mb * num_clients) / 5  # 5 MB/s estimate
        streaming_timeout = base_timeout + int(transfer_time * 2)

        # Timeouts for large models with many clients
        cell_timeout = 60 if model_size_mb > 100 else 30
        task_check_timeout = 30 if num_clients > 4 else 10
        runner_sync_timeout = 10 if num_clients > 4 else 5

        print(f"[Config] streaming_per_request_timeout: {streaming_timeout}s (default: 600s)")
        print(f"[Config] cell_wait_timeout: {cell_timeout}s (default: 5s)")
        print(f"[Config] task_check_timeout: {task_check_timeout}s (default: 5s)")

        recipe.add_server_config({
            "streaming_per_request_timeout": streaming_timeout,
            "cell_wait_timeout": cell_timeout,
        })
        recipe.add_client_config({
            "cell_wait_timeout": cell_timeout,
            "task_check_timeout": task_check_timeout,
            "runner_sync_timeout": runner_sync_timeout,
        })

        t3 = time.time()
        print(f"[Timing] Create FedAvgRecipe: {t3 - t2:.2f}s")

        # Stage 4: Create SimEnv (num_threads = num_clients for parallel execution)
        env = SimEnv(num_clients=num_clients, num_threads=num_clients)
        t4 = time.time()
        print(f"[Timing] Create SimEnv: {t4 - t3:.2f}s")

        # Stage 5: Run the simulation
        print("[Timing] Starting recipe.execute()...")
        run = recipe.execute(env)
        t5 = time.time()
        print(f"[Timing] recipe.execute() completed: {t5 - t4:.2f}s")
        print(f"[Timing] Total time: {t5 - t0:.2f}s")

    # Force GC after run
    gc.collect()
    final_rss = get_rss_mb()
    peak_rss = get_peak_rss_mb()
    rss_increase = final_rss - initial_rss

    print(f"Final RSS: {final_rss:.1f} MB")
    print(f"Peak RSS: {peak_rss:.1f} MB")
    print(f"RSS increase: {rss_increase:.1f} MB")

    return {
        "server_memory_gc_rounds": server_memory_gc_rounds,
        "peak_rss_mb": peak_rss,
        "initial_rss_mb": initial_rss,
        "final_rss_mb": final_rss,
        "rss_increase_mb": rss_increase,
    }


def print_summary(results: list):
    """Print summary comparison of all test runs.

    Args:
        results: List of result dictionaries from run_simulation.
    """
    line_width = 85
    print(f"\n{'=' * line_width}")
    print("SUMMARY")
    print(f"{'=' * line_width}")
    print(f"{'Setting':<25} {'Initial MB':>12} {'Peak MB':>12} {'Final MB':>12} {'Increase':>12}")
    print("-" * line_width)

    for r in results:
        gc_setting = r["server_memory_gc_rounds"]
        label = f"gc_rounds={gc_setting}" if gc_setting > 0 else "gc_rounds=0 (disabled)"
        peak = r.get("peak_rss_mb", r["final_rss_mb"])  # fallback if not present
        print(
            f"{label:<25} "
            f"{r['initial_rss_mb']:>12.1f} "
            f"{peak:>12.1f} "
            f"{r['final_rss_mb']:>12.1f} "
            f"{r['rss_increase_mb']:>+12.1f}"
        )

    print("-" * line_width)

    # Calculate improvement
    if len(results) >= 2:
        baseline = results[0]["rss_increase_mb"]
        for r in results[1:]:
            if baseline > 0:
                improvement = (baseline - r["rss_increase_mb"]) / baseline * 100
                print(f"gc_rounds={r['server_memory_gc_rounds']} vs disabled: {improvement:.1f}% reduction")


def run_single_test(
    gc_rounds: int, num_rounds: int, num_clients: int, model_size_mb: int, output_json: bool = False
):
    """Run a single test and print result.

    Args:
        gc_rounds: server_memory_gc_rounds value.
        num_rounds: Number of FL rounds.
        num_clients: Number of clients.
        model_size_mb: Model size in MB.
        output_json: If True, output JSON for subprocess parsing. If False, show table.
    """
    result = run_simulation(
        server_memory_gc_rounds=gc_rounds,
        num_rounds=num_rounds,
        num_clients=num_clients,
        model_size_mb=model_size_mb,
    )
    if output_json:
        # JSON output for subprocess parsing
        print(f"__RESULT_JSON__:{json.dumps(result)}")
    else:
        # Nice table format for direct run
        print_summary([result])


def run_tests_in_subprocess(gc_rounds_list: list, num_rounds: int, num_clients: int, model_size_mb: int) -> list:
    """Run each test in a separate subprocess for complete isolation.

    Args:
        gc_rounds_list: List of server_memory_gc_rounds values to test.
        num_rounds: Number of FL rounds per test.
        num_clients: Number of simulated clients.
        model_size_mb: Model size in MB.

    Returns:
        List of result dictionaries.
    """
    results = []
    script_path = os.path.abspath(__file__)

    for gc_rounds in gc_rounds_list:
        print(f"\n{'=' * 60}")
        print(f"Starting subprocess for gc_rounds={gc_rounds}")
        print(f"{'=' * 60}")

        cmd = [
            sys.executable,
            script_path,
            "--single",
            str(gc_rounds),
            "--json",  # Output JSON for parsing
            "--num-rounds",
            str(num_rounds),
            "--num-clients",
            str(num_clients),
            "--model-size",
            str(model_size_mb),
        ]

        # Run subprocess and capture output
        proc = subprocess.run(cmd, capture_output=True, text=True)

        # Print subprocess output
        if proc.stdout:
            for line in proc.stdout.splitlines():
                if line.startswith("__RESULT_JSON__:"):
                    # Parse the JSON result
                    json_str = line.replace("__RESULT_JSON__:", "")
                    result = json.loads(json_str)
                    results.append(result)
                else:
                    print(line)

        if proc.stderr:
            # Filter out common warnings
            for line in proc.stderr.splitlines():
                if "FutureWarning" not in line and "pynvml" not in line:
                    print(f"[stderr] {line}", file=sys.stderr)

        if proc.returncode != 0:
            print(f"[WARNING] Subprocess exited with code {proc.returncode}")

    return results


def main():
    """Run memory profiling tests."""
    parser = argparse.ArgumentParser(description="FedAvg Memory Profiling Test")
    parser.add_argument(
        "--single", type=int, metavar="GC_ROUNDS", help="Run single test with specified gc_rounds"
    )
    parser.add_argument("--json", action="store_true", help="Output JSON (for subprocess mode)")
    parser.add_argument("--num-rounds", type=int, default=10, help="Number of FL rounds (default: 10)")
    parser.add_argument("--num-clients", type=int, default=2, help="Number of clients (default: 2)")
    parser.add_argument("--model-size", type=int, default=100, help="Model size in MB (default: 100)")
    args = parser.parse_args()

    if args.single is not None:
        # Single test mode
        run_single_test(
            gc_rounds=args.single,
            num_rounds=args.num_rounds,
            num_clients=args.num_clients,
            model_size_mb=args.model_size,
            output_json=args.json,
        )
    else:
        # Main mode: run all tests in subprocesses
        print("FedAvg Memory Profiling Test")
        print(f"MALLOC_ARENA_MAX: {os.environ.get('MALLOC_ARENA_MAX', 'not set')}")
        print(f"Python: {sys.version}")
        print("\nEach test runs in a separate subprocess for complete isolation.")

        # Test configurations
        gc_rounds_list = [0, 5, 1]

        results = run_tests_in_subprocess(
            gc_rounds_list=gc_rounds_list,
            num_rounds=args.num_rounds,
            num_clients=args.num_clients,
            model_size_mb=args.model_size,
        )

        if results:
            print_summary(results)
        else:
            print("\n[ERROR] No results collected. Check subprocess output above.")


if __name__ == "__main__":
    main()
