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

"""Evaluate an exported VLA checkpoint; federation runs through FLARE commands."""

import argparse
import dataclasses
import logging
import time
from pathlib import Path

from decent_vla._helpers.builders import build_backend_factory, resolve_device
from decent_vla._helpers.eval_runner import run_eval
from decent_vla._helpers.personalized_eval import run_personalized_eval
from decent_vla._helpers.results import now_iso
from decent_vla.configs import load_run_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", help="evaluation output folder; default <run>/eval")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_run_config(args.config)
    if cfg.eval is None:
        parser.error("the run YAML needs an eval block")
    if not args.ckpt.is_file():
        parser.error(f"checkpoint does not exist: {args.ckpt}")
    run_dir = args.ckpt.resolve().parent.parent
    cfg.eval.output_dir = args.output_dir or cfg.eval.output_dir or str(run_dir / "eval")
    started, t0 = now_iso(), time.time()
    factory, _ = build_backend_factory(cfg, resolve_device(cfg.device))
    if cfg.eval.personalized:
        result = run_personalized_eval(cfg, factory, run_dir, cfg.eval.output_dir, ckpt_path=args.ckpt)
    else:
        result = run_eval(cfg, factory(), args.ckpt, run_dir, started, t0, dataclasses.asdict(cfg), args.config)
    print(result)


if __name__ == "__main__":
    main()
