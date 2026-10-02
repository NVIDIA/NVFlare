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

import argparse
import os
from datetime import datetime

import torch
from easydict import EasyDict

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_LOAD_DIR = os.path.join(PROJECT_DIR, "dump")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "dump")
DEFAULT_DATA_DIR = os.path.join(PROJECT_DIR, "data")
DEFAULT_RAW_DIR = os.path.join(DEFAULT_DATA_DIR, "raw")
DEFAULT_MANIFEST_PATH = os.path.join(PROJECT_DIR, "metadata", "gdc_files_manifest.tsv")
DEFAULT_CLINICAL_PATH = os.path.join(PROJECT_DIR, "metadata", "gdc_clinical_cases.tsv")
DEFAULT_TCIA_SPATIAL_FEATURES_PATH = os.path.join(
    DEFAULT_DATA_DIR,
    "tcia_official_radiogenomics",
    "official_spatial_patient_features.csv",
)
DEFAULT_DCE_KINETICS_PATH = os.path.join(
    DEFAULT_DATA_DIR,
    "tcia_official_radiogenomics",
    "official_temporal_patient_features.csv",
)


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def build_parser():
    parser = argparse.ArgumentParser(description="TCGA-BRCA GFedCL configuration")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", type=_str2bool, default=False)

    parser.add_argument("--load-dir", default=DEFAULT_LOAD_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-path", default=None)

    parser.add_argument("--dataset", default="TCGA-BRCA")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--clinical-path", default=DEFAULT_CLINICAL_PATH)
    parser.add_argument(
        "--task-split-strategy",
        default="clinical_stage",
        choices=["clinical_stage", "random"],
        help="FCL task semantics. clinical_stage uses AJCC stage progression tasks.",
    )
    parser.add_argument("--include-unknown-stage", type=_str2bool, default=False)
    parser.add_argument("--expression-value-col", default="tpm_unstranded")
    parser.add_argument("--tcia-mri-features-path", default=DEFAULT_TCIA_SPATIAL_FEATURES_PATH)
    parser.add_argument("--tcia-mri-id-column", default=None)
    parser.add_argument("--tcia-dce-kinetics-path", default=DEFAULT_DCE_KINETICS_PATH)
    parser.add_argument("--tcia-dce-id-column", default=None)
    parser.add_argument("--max-genes", type=int, default=4096)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--pin-memory", type=_str2bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--test-split", type=float, default=0.2)

    parser.add_argument("--use-g-encode", type=_str2bool, default=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-gan", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-d", type=float, default=1e-4)
    parser.add_argument("--lr-f", type=float, default=1e-4)
    parser.add_argument("--lr-e", type=float, default=1e-4)
    parser.add_argument("--lr-g", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1000)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--shuffle", type=_str2bool, default=True)

    parser.add_argument("--gat-rounds", type=int, default=10)
    parser.add_argument(
        "--gat-epochs",
        type=int,
        default=20,
        help="Compatibility option; attention currently runs inference without separate training.",
    )
    parser.add_argument(
        "--gat-lr",
        type=float,
        default=1e-5,
        help="Reserved for attention training; no separate attention optimizer is configured.",
    )
    parser.add_argument("--gat-hidden-dim", type=int, default=128)
    parser.add_argument("--gat-embedding-dim", type=int, default=64)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--gat-dropout", type=float, default=0.2)
    parser.add_argument("--temporal-window", type=int, default=2)
    parser.add_argument("--attention-temperature", type=float, default=1.0)
    parser.add_argument("--graph-epsilon", type=float, default=1e-8)

    parser.add_argument("--num-task", type=int, default=3)
    parser.add_argument("--class-per-task", type=int, default=2)
    parser.add_argument("--num-local-epochs", type=int, default=20)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--num-clients", type=int, default=4)

    parser.add_argument(
        "--nt",
        type=int,
        default=None,
        help="Compatibility option; graph embedding dimension is set by --num-clients.",
    )
    parser.add_argument("--nh", type=int, default=800)
    parser.add_argument(
        "--noise-dim", type=int, default=100, help="Noise width for the graph-conditioned latent replay generator."
    )
    parser.add_argument("--ni", type=int, default=800)
    parser.add_argument("--nc", type=int, default=2)
    parser.add_argument(
        "--nd-out",
        type=int,
        default=None,
        help="Compatibility option; discriminator output dimension is set by --num-clients.",
    )
    parser.add_argument("--p", type=float, default=0.2)
    parser.add_argument("--no-bn", type=_str2bool, default=True)

    parser.add_argument("--sensitivity", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1.0)

    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        help="Maximum simultaneous NVFlare client operations (default: min(num_clients, 8)).",
    )

    parser.add_argument("--replay", type=_str2bool, default=True)

    return parser


def finalize_opt(opt):
    opt.train = not opt.debug
    if opt.device == "cuda" and not torch.cuda.is_available():
        opt.device = "cpu"

    if opt.output_dir == DEFAULT_OUTPUT_DIR:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        opt.output_dir = os.path.join(opt.output_dir, timestamp)
        opt.output_dir_is_default = True
    else:
        opt.output_dir_is_default = False

    os.makedirs(opt.load_dir, exist_ok=True)
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(opt.data_dir, exist_ok=True)
    os.makedirs(opt.raw_dir, exist_ok=True)

    if opt.log_path is None:
        opt.log_path = os.path.join(opt.output_dir, "run.log")

    opt.b = opt.sensitivity / opt.epsilon if opt.epsilon != 0 else 0.0
    opt.nc = opt.num_classes
    opt.ni = opt.nh
    opt.nt = opt.num_clients
    opt.nd_out = opt.nt

    if opt.max_in_flight is None:
        opt.max_in_flight = min(opt.num_clients, 8)
    if opt.max_in_flight < 1:
        raise ValueError("--max-in-flight must be positive")

    return opt


def parse_args(args=None):
    parser = build_parser()
    parsed = parser.parse_args(args=args)
    opt = EasyDict(vars(parsed))
    return finalize_opt(opt)


opt = parse_args(args=[])
