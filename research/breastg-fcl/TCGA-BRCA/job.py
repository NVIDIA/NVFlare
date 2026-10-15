# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# The original BreastG-FCL MIT notice is retained below for the upstream code.
# MIT License
#
# Copyright (c) 2026 IntelliSys-Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Build and run the BreastG-FCL CollabRecipe with isolated site data."""

import argparse
import gc
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from federated.client import BreastGFCLClient
from federated.server import BreastGFCLServer
from prepare_data import prepare_bundles

from nvflare.collab import CollabRecipe
from nvflare.recipe import SimEnv

PROJECT = Path(__file__).resolve().parent
JOB_NAME = "breastg_fcl"


def make_recipe(server_bundle_path, site_bundle_paths):
    """Connect the published client operations to the server's continual workflow."""
    site_bundle_paths = list(site_bundle_paths)
    if not site_bundle_paths:
        raise ValueError("At least one client bundle is required")
    recipe = CollabRecipe(
        job_name=JOB_NAME,
        server=BreastGFCLServer(),
        client=BreastGFCLClient(),
        min_clients=len(site_bundle_paths),
        max_call_threads_for_client=1,
    )
    # Separate client apps keep every raw split confined to its assigned site.
    recipe.set_per_site_config({f"site-{i + 1}": {"client_id": i} for i in range(len(site_bundle_paths))})
    job = recipe.finalize()
    targets = ["server", *(f"site-{i + 1}" for i in range(len(site_bundle_paths)))]
    for target, bundle_path in zip(targets, [server_bundle_path, *site_bundle_paths]):
        job.add_file_to(str(bundle_path), target, dest_dir="data", app_folder_type="config")
        # Include only the modules required by the workflow, never source data or outputs.
        for package in ("model", "utils", "configs", "federated"):
            for source in sorted((PROJECT / package).glob("*.py")):
                job.add_file_to(str(source), target, dest_dir=package, app_folder_type="custom")
        job.add_file_to(str(PROJECT / "breastgfcl.py"), target, app_folder_type="custom")
    return recipe


def export_job(job_dir, server_bundle, site_bundles):
    """Let CollabRecipe generate the apps and configuration for a prepared split."""
    job_dir = Path(job_dir)
    if job_dir.exists():
        raise FileExistsError(f"Job directory already exists: {job_dir}")
    with tempfile.TemporaryDirectory(prefix="breastgfcl-bundles-") as directory:
        staging = Path(directory)
        server_path = staging / "server.pt"
        torch.save(server_bundle, server_path)
        site_paths = []
        for index, bundle in enumerate(site_bundles):
            site_path = staging / f"site-{index + 1}" / "site.pt"
            site_path.parent.mkdir()
            torch.save(bundle, site_path)
            site_paths.append(site_path)
        make_recipe(server_path, site_paths).export(str(job_dir))
    return job_dir / JOB_NAME


def configure_cuda(device, gpu):
    """Select physical GPUs before importing configuration or querying CUDA.

    Without an explicit selection, preserve the inherited mask and ordering;
    NVFlare workers must then inherit them too, including UUID/MIG masks.
    """
    if device not in (None, "cpu", "cuda"):
        raise ValueError("--device must be 'cpu' or 'cuda'; select physical GPUs with --nvflare-gpu")
    if torch.cuda.is_initialized():
        raise ValueError("CUDA is already initialized; launch job.py in a fresh interpreter to select devices safely")
    if device == "cpu":
        if gpu is not None:
            raise ValueError("--nvflare-gpu requires --device cuda")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return None
    if gpu is None:
        return None
    gpu = gpu.replace(" ", "")
    if re.fullmatch(r"[0-9]+|\[[0-9]+(?:,[0-9]+)*\]", gpu) is None:
        raise ValueError("--nvflare-gpu must select one GPU or one shared GPU group, e.g. '1' or '[1,2]'")
    selected = [str(int(value)) for value in gpu.strip("[]").split(",")]
    if len(set(selected)) != len(selected):
        raise ValueError("--nvflare-gpu must not contain duplicate GPU IDs")
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None:
        allowed = inherited.replace(" ", "").split(",")
        if (
            os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
            or not all(value.isdecimal() for value in allowed)
            or not set(selected).issubset({str(int(value)) for value in allowed})
        ):
            raise ValueError(
                "--nvflare-gpu conflicts with or cannot safely interpret inherited CUDA_VISIBLE_DEVICES; "
                "omit --nvflare-gpu to preserve the inherited selection, or use a compatible numeric mask "
                "with CUDA_DEVICE_ORDER=PCI_BUS_ID"
            )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
    return selected[0] if len(selected) == 1 else "[" + ",".join(selected) + "]"


def validate_simulator_options(opt):
    """Keep each Collab site's initialized object in a persistent worker."""
    if opt.num_clients <= 0:
        raise ValueError("--num-clients must be positive")
    if opt.nvflare_threads is not None:
        if opt.nvflare_threads <= 0:
            raise ValueError("--nvflare-threads must be positive")
        if opt.nvflare_threads != opt.num_clients:
            raise ValueError(
                "--nvflare-threads must equal --num-clients so Collab workers remain active; "
                "use --max-in-flight to limit training concurrency"
            )
    if opt.nvflare_gpu is not None:
        if opt.device == "cpu":
            raise ValueError("--nvflare-gpu requires --device cuda")
        # Multiple GPU groups make the 2.9 simulator force one worker per group
        # and rotate its clients, losing Collab objects and their RPC handlers.
        gpu = opt.nvflare_gpu.replace(" ", "")
        if re.fullmatch(r"[0-9]+|\[[0-9]+(?:,[0-9]+)*\]", gpu) is None:
            raise ValueError(
                "--nvflare-gpu must select one GPU or one shared GPU group, e.g. '0' or '[0,1]'; "
                "multiple GPU groups are not supported by this Collab simulator launcher"
            )


def execute_exported_job(opt):
    """Run the exported data in a fresh interpreter using the public Recipe API."""
    validate_simulator_options(opt)
    job_dir = Path(opt.run_exported_job).resolve()
    server_path = job_dir / "app_server" / "config" / "data" / "server.pt"
    site_paths = [job_dir / f"app_site-{i + 1}" / "config" / "data" / "site.pt" for i in range(opt.num_clients)]
    recipe = make_recipe(server_path, site_paths)
    gpu = opt.nvflare_gpu
    # SimEnv owns job deployment and execution; each Collab call carries native tensors.
    run = recipe.execute(
        SimEnv(
            num_clients=opt.num_clients,
            clients=[f"site-{i + 1}" for i in range(opt.num_clients)],
            num_threads=opt.nvflare_threads or opt.num_clients,
            gpu_config=gpu,
            workspace_root=opt.nvflare_workspace,
        )
    )
    print("Job status:", run.get_status(), flush=True)
    print("Workspace:", run.get_result(), flush=True)


def run_simulator(opt, job_dir, output):
    # Data/model preparation can initialize CUDA and autograd threads. Start the
    # Recipe in a new interpreter before SimEnv creates its server and clients.
    validate_simulator_options(opt)
    workspace = str(Path(opt.nvflare_workspace or output / "nvflare_workspace").resolve())
    command = [
        sys.executable,
        str(PROJECT / "job.py"),
        "--run-exported-job",
        str(job_dir),
        "--num-clients",
        str(opt.num_clients),
        "--device",
        opt.device,
        "--nvflare-workspace",
        workspace,
        "--nvflare-threads",
        str(opt.nvflare_threads or opt.num_clients),
    ]
    if opt.nvflare_gpu is not None:
        command.extend(["--nvflare-gpu", opt.nvflare_gpu])
    log_path = output / "simulator.log"
    print(f"Running CollabRecipe; simulator log: {log_path}", flush=True)
    with log_path.open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not (output / "final_state.pt").exists():
        raise RuntimeError(
            f"NVFlare failed or produced no final state (exit code {result.returncode}); "
            f"inspect {log_path} and {workspace}"
        )


def main(args=None):
    # The configuration module queries CUDA at import time. Resolve visibility
    # first, before that query can lock the runtime to the wrong physical GPU.
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("--device", default=None)
    bootstrap.add_argument("--nvflare-gpu", default=None)
    selection, _ = bootstrap.parse_known_args(args)
    try:
        gpu = configure_cuda(selection.device, selection.nvflare_gpu)
    except ValueError as error:
        bootstrap.error(str(error))
    from configs.TCGA_BRCA import build_parser, finalize_opt

    parser = build_parser()
    parser.allow_abbrev = False
    parser.description = "BreastG-FCL federated continual learning with the NVFlare Collab API"
    parser.add_argument("--smoke", action="store_true", help="Use a small synthetic validation fixture")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--nvflare-workspace", default=None)
    parser.add_argument("--nvflare-threads", type=int, default=None, help="Must equal the client count (default)")
    parser.add_argument(
        "--nvflare-gpu",
        default=None,
        help="Physical GPU ID or shared group for preparation, server and clients; defaults to inherited visibility",
    )
    parser.add_argument("--nvflare-timeout", type=int, default=300)
    parser.add_argument("--run-exported-job", help=argparse.SUPPRESS)
    args = parser.parse_args(args)
    args.nvflare_gpu = gpu
    if args.nvflare_timeout <= 0:
        parser.error("--nvflare-timeout must be positive")
    try:
        validate_simulator_options(args)
        if gpu is not None and not torch.cuda.is_available():
            raise ValueError("--nvflare-gpu selected no available CUDA device")
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    if args.run_exported_job:
        execute_exported_job(args)
        return
    opt = finalize_opt(args)
    if opt.smoke:
        opt.input_dim, opt.nh, opt.ni, opt.noise_dim = 8, 16, 16, 5
        opt.batch_size, opt.num_local_epochs, opt.num_rounds = 4, 1, 2
        opt.gat_hidden_dim, opt.gat_embedding_dim, opt.gat_heads = 32, 16, 2
        opt.no_bn, opt.p = False, 0.1
    output = Path(opt.output_dir).resolve()
    opt.output_dir = str(output)
    workflow, bundle, sites = prepare_bundles(opt, opt.smoke)
    job_dir = export_job(output / "nvflare_job", bundle, sites)
    print(f"Exported Collab job: {job_dir}", flush=True)
    if opt.export_only:
        return
    del workflow, bundle, sites
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    run_simulator(opt, job_dir, output)
    print(f"NVFlare results: {output}", flush=True)


if __name__ == "__main__":
    main()
