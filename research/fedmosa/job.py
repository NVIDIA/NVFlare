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

"""Builds and runs the FedMoSA job in the NVFlare simulator."""

import argparse
import os
import shlex

from src.fedmosa_controller import FedMoSAController
from src.model import MoSA

from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner

HERE = os.path.dirname(os.path.abspath(__file__))


class InitialModel(MoSA):
    """Server-side initial model; only the trainable (exchanged) parameters are put in the global model."""

    def state_dict(self, *args, **kwargs):
        return {k: v for k, v in super().state_dict(*args, **kwargs).items() if k in self.trainable_state()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dataset folder with slices_*.pkl (system/prepare_data.py)")
    ap.add_argument("--dataset-id", default="0")
    ap.add_argument("--sites", default="site-1,site-2,site-3")
    ap.add_argument("--modalities", default="MR,CT")
    ap.add_argument("--sam-ckpt", default="", help="SAM ViT-B checkpoint (sam_vit_b_01ec64.pth or MedSAM)")
    ap.add_argument(
        "--decoder-init",
        default="random",
        choices=["random", "pretrained"],
        help="mask decoder: trained from a random initialization (default) or from the SAM checkpoint",
    )
    ap.add_argument("--num-rounds", type=int, default=50)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam-proto", type=float, default=0.01)
    ap.add_argument("--workspace", default="/tmp/nvflare/fedmosa")
    ap.add_argument("--gpu", default="")
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="simulator threads (default: one per site, so each client keeps its process and optimizer state)",
    )
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()

    sites = args.sites.split(",")
    mods = args.modalities.split(",")
    job = FedJob(name="fedmosa")
    ckpt = os.path.abspath(args.sam_ckpt) if args.sam_ckpt else None  # absolute: the server runs in its workspace
    comp_ids = job.to_server(PTModel(InitialModel(ckpt, mods, decoder_init=args.decoder_init)))
    job.to_server(
        FedMoSAController(
            num_clients=len(sites),
            num_rounds=args.num_rounds,
            modalities=mods,
            persistor_id=comp_ids["persistor_id"],
        ),
        id="fedmosa",
    )

    script_args = [
        "--data-dir", os.path.abspath(args.data_dir),
        "--dataset-id", args.dataset_id,
        "--sites", args.sites,
        "--modalities", args.modalities,
        "--local-epochs", str(args.local_epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--lam-proto", str(args.lam_proto),
        "--decoder-init", args.decoder_init,
    ]  # fmt: skip
    if ckpt:
        script_args += ["--sam-ckpt", ckpt]
    job.to_clients(os.path.join(HERE, "src"))  # ship src/ (model, data, client script) with the client app
    job.to_clients(ScriptRunner(script="src/client.py", script_args=shlex.join(script_args)))

    if args.export_only:
        job.export_job(os.path.join(args.workspace, "jobs"))
    else:
        job.simulator_run(args.workspace, clients=sites, threads=args.threads or len(sites), gpu=args.gpu or None)


if __name__ == "__main__":
    main()
