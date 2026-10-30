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

"""Export a decent-vla server checkpoint as a lerobot-native pretrained dir.

`server_round_*.pt` holds only the FL shared state (LoRA adapters + trainable
head params). `lerobot-eval --policy.path=...` wants a self-contained
`pretrained_model/` dir with merged full weights. This module builds it: load shared state into the backend's policy, `merge_and_unload()`
every PEFT wrapper back into its parent module, then write the checkpoint
through lerobot's own `save_checkpoint`.

Usage (CPU is fine — it's weight surgery, not training):

    python -m decent_vla.adapters.lerobot.export \
        --config outputs/smolvla_libero_lora_nvflare/run.yaml \
        --ckpt outputs/smolvla_libero_lora_nvflare/checkpoints/server_round_final.pt \
        --output-dir outputs/smolvla_libero_lora_nvflare/export
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def make_export_portable(out_dir) -> None:
    """Post-process a freshly-saved lerobot policy dir so it loads + runs on a
    fresh deployment target (e.g. a Jetson Orin via ``lerobot-record``), undoing
    two things decent-vla's offline training bakes in:

      1. The paligemma tokenizer redirected to a CLUSTER-LOCAL HF snapshot path
         (``.../hub/models--google--paligemma-3b-pt-224/snapshots/<hash>``) that
         exists nowhere else -> restore the portable repo id
         (``models--A--B/snapshots/...`` -> ``A/B``).
      2. pi05 pads state/action to ``max_*_dim`` (32); the saved config then
         advertises the PADDED dims, so native inference feeds the model's 32-d
         action into the real-D (e.g. 6) unnormalizer and crashes. Reset the
         config ``observation.state`` / ``action`` feature shapes to the REAL
         dims read from the saved normalization stats, so the model slices its
         padded output back to the robot's action space.

    No-op when already portable (repo-id tokenizer, native-dim config), so it is
    safe for every backend (smolvla/groot/act configs are untouched).
    """
    import glob
    import json
    import os
    import re

    out_dir = str(out_dir)

    pp = os.path.join(out_dir, "policy_preprocessor.json")
    if os.path.isfile(pp):
        blob = json.load(open(pp))
        changed = False

        def _walk(o):
            nonlocal changed
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == "tokenizer_name" and isinstance(v, str) and os.path.isabs(v):
                        m = re.search(r"models--([^/]+)/snapshots", v)
                        if m:
                            o[k] = m.group(1).replace("--", "/")
                            changed = True
                    else:
                        _walk(v)
            elif isinstance(o, list):
                for x in o:
                    _walk(x)

        _walk(blob)
        if changed:
            json.dump(blob, open(pp, "w"), indent=2)
            logger.info("export: tokenizer_name -> portable repo id")

    cp = os.path.join(out_dir, "config.json")
    if os.path.isfile(cp):
        from safetensors.torch import load_file

        def _stat_dim(pattern, key):
            for f in glob.glob(os.path.join(out_dir, pattern)):
                for kk, vv in load_file(f).items():
                    if key in kk and ("mean" in kk or "std" in kk):
                        return int(vv.numel())
            return None

        cfg = json.load(open(cp))
        touched = False
        a = _stat_dim("*postprocessor*unnormalizer*.safetensors", "action")
        s = _stat_dim("*preprocessor*normalizer*.safetensors", "observation.state")
        of = cfg.get("output_features", {})
        inf = cfg.get("input_features", {})
        if a and "action" in of and of["action"].get("shape") != [a]:
            of["action"]["shape"] = [a]
            touched = True
        if s and "observation.state" in inf and inf["observation.state"].get("shape") != [s]:
            inf["observation.state"]["shape"] = [s]
            touched = True
        if touched:
            json.dump(cfg, open(cp, "w"), indent=2)
            logger.info("export: config feature dims -> real (action=%s, state=%s)", a, s)


def export_pretrained(backend, ckpt_path: str | Path, output_dir: str | Path) -> Path:
    """Materialize `ckpt_path` (a server checkpoint) as
    `{output_dir}/round_{NNNN}/pretrained_model/`; returns that path."""
    try:
        from lerobot.utils.train_utils import save_checkpoint, update_last_checkpoint
    except ImportError:  # lerobot moved train_utils to lerobot.common on 2026-04-12 (#3362)
        from lerobot.common.train_utils import save_checkpoint, update_last_checkpoint
    from peft import PeftModel

    ckpt_path = Path(ckpt_path)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    step = int(blob.get("round", -1)) + 1 if isinstance(blob, dict) else 0

    backend.load_checkpoint(str(ckpt_path))
    policy = backend.model

    # Fold every PEFT wrapper into its base and splice the merged module back
    # where the wrapper sat.
    merged_any = False
    for name, module in list(policy.named_modules()):
        if isinstance(module, PeftModel) and name:
            merged = module.merge_and_unload()
            parent = policy
            *parents, leaf = name.split(".")
            for p in parents:
                parent = getattr(parent, p)
            setattr(parent, leaf, merged)
            merged_any = True
            logger.info("merged PEFT wrapper at %r", name)
    if not merged_any:
        logger.info("no PEFT wrapper found — exporting weights as-is")

    # Persist an inference device in the saved config. This export runs
    # CPU-only (weight surgery, not training), so policy.config.device would
    # otherwise be "cpu" and downstream lerobot-eval / from_pretrained would
    # place the VLM on the CPU. Force "cuda".
    if getattr(getattr(policy, "config", None), "device", None) is not None:
        policy.config.device = "cuda"

    checkpoint_dir = Path(output_dir) / f"round_{step:04d}"
    save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        step=step,
        cfg=backend._train_pipeline_cfg,
        policy=policy,
        optimizer=None,
        scheduler=None,
        preprocessor=backend._preprocessor,
        postprocessor=backend._postprocessor,
    )
    update_last_checkpoint(checkpoint_dir)
    out = checkpoint_dir / "pretrained_model"
    make_export_portable(out)
    logger.info("exported %s (round %d) -> %s", ckpt_path.name, step, out)
    return out


def main() -> int:
    import argparse

    from decent_vla.configs.loader import load_run_config
    from decent_vla.runtime.builders import build_backend_factory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="decent-vla run config YAML")
    parser.add_argument("--ckpt", required=True, help="server_round_*.pt to export")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = load_run_config(args.config)
    factory, _ = build_backend_factory(cfg, torch.device("cpu"))
    backend = factory()
    out = export_pretrained(backend, args.ckpt, args.output_dir)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
