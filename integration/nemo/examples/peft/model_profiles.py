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
"""Named model profiles for the NeMo AutoModel federated PEFT example."""

from __future__ import annotations

from copy import deepcopy

NANO_PROFILE = "nano"
LIGHTNING35_PROFILE = "lightning35"

PROFILES = {
    NANO_PROFILE: {
        "model_name_or_path": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "tokenizer_name_or_path": None,
        "initial_adapter_ckpt": "./models/nemotron3_nano_lora_init.pt",
        "workspace": "/tmp/nvflare/nemotron3_nano_peft",
        "learning_rate": 2e-4,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": "all-linear",
        "exclude_modules": "",
        "use_triton_lora": False,
        "tp_size": 1,
        "cp_size": 1,
        "ep_size": 1,
        "activation_checkpointing": False,
    },
    LIGHTNING35_PROFILE: {
        "model_name_or_path": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "tokenizer_name_or_path": None,
        "initial_adapter_ckpt": "./models/nemotron35_lightning_lora_init.pt",
        "workspace": "/tmp/nvflare/nemotron35_lightning_peft",
        "learning_rate": 5e-5,
        "lora_rank": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "target_modules": "all-linear",
        "exclude_modules": "*.out_proj",
        "use_triton_lora": True,
        "tp_size": 1,
        "cp_size": 1,
        "ep_size": 1,
        "activation_checkpointing": False,
    },
}

NATIVE_MODEL_SETTINGS = {
    NANO_PROFILE: {},
    LIGHTNING35_PROFILE: {
        "backend": {
            "attn": "te",
            "linear": "torch",
            "rms_norm": "torch_fp32",
            "experts": "torch_mm",
            "dispatcher": "torch",
        },
        "num_nextn_predict_layers": 2,
        "mtp_use_repeated_layer": True,
        "mtp_loss_scaling_factor": 0.1,
    },
}


def add_model_profile_argument(parser) -> None:
    parser.add_argument(
        "--model_profile",
        choices=tuple(PROFILES),
        default=NANO_PROFILE,
        help="Named defaults; explicit command-line training options take precedence.",
    )


def resolve_model_profile(args):
    """Fill only unspecified profile-aware arguments and return ``args``."""
    profile_name = getattr(args, "model_profile", NANO_PROFILE)
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown model profile: {profile_name}")
    args.model_profile = profile_name
    values = deepcopy(PROFILES[profile_name])
    for name, value in values.items():
        if not hasattr(args, name) or getattr(args, name) is None:
            setattr(args, name, value)
    if not args.tokenizer_name_or_path:
        args.tokenizer_name_or_path = args.model_name_or_path
    if not hasattr(args, "model_revision"):
        args.model_revision = None
    if not hasattr(args, "tokenizer_revision"):
        args.tokenizer_revision = None
    return args


def profile_defaults(profile_name: str) -> dict:
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown model profile: {profile_name}")
    return deepcopy(PROFILES[profile_name])


def is_lightning35(args) -> bool:
    return getattr(args, "model_profile", NANO_PROFILE) == LIGHTNING35_PROFILE


def native_model_settings(args) -> dict:
    """Return native model construction settings for the selected profile."""
    profile_name = getattr(args, "model_profile", NANO_PROFILE)
    if profile_name not in NATIVE_MODEL_SETTINGS:
        raise ValueError(f"Unknown model profile: {profile_name}")
    return deepcopy(NATIVE_MODEL_SETTINGS[profile_name])


def adapter_compatibility_settings(args) -> dict:
    """Return settings that must remain identical when a native adapter is loaded."""
    settings = {
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "exclude_modules": args.exclude_modules,
        "use_triton_lora": args.use_triton_lora,
        "tp_size": args.tp_size,
        "cp_size": args.cp_size,
        "ep_size": args.ep_size,
    }
    settings.update(native_model_settings(args))
    return settings


def adapter_identity(args) -> dict:
    """Return the immutable model, tokenizer, and profile identity for an adapter."""
    args = resolve_model_profile(args)
    return {
        "model_profile": args.model_profile,
        "base_model_name_or_path": args.model_name_or_path,
        "base_model_revision": args.model_revision,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "tokenizer_revision": args.tokenizer_revision,
        "profile_settings": adapter_compatibility_settings(args),
    }


def default_server_model_path(args) -> str:
    """Return the default final server checkpoint path for the selected profile."""
    args = resolve_model_profile(args)
    job_name = "nemotron35-lightning-peft" if is_lightning35(args) else "nemotron3-nano-peft"
    return f"{args.workspace}/{job_name}/server/simulate_job/app_server/FL_global_model.pt"
