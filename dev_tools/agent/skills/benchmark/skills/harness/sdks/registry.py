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

"""SDK adapter registry for benchmark Docker setup."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .base import SdkAdapter
from .config import ConfigurableSdkAdapter

BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
SDK_CONFIG_DIR = BENCHMARK_ROOT / "config" / "sdks"
DEFAULT_BENCHMARK_SDK = "nvflare-profile"


def sdk_config_path(sdk: str) -> Path:
    if Path(sdk).name != sdk or sdk.startswith("."):
        raise ValueError(f"SDK profile must be a config name, got {sdk!r}")
    return SDK_CONFIG_DIR / f"{sdk}.yaml"


def supported_sdk_names() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in SDK_CONFIG_DIR.glob("*.yaml") if not path.name.startswith(".")))


def validate_benchmark_sdk(sdk: str) -> str:
    config_path = sdk_config_path(sdk)
    if not config_path.is_file():
        supported = ", ".join(supported_sdk_names())
        raise ValueError(f"Unsupported SDK profile {sdk!r}. Supported SDK profiles: {supported}.")
    return sdk


@lru_cache(maxsize=None)
def get_sdk_adapter(sdk: str) -> SdkAdapter:
    name = validate_benchmark_sdk(sdk)
    return ConfigurableSdkAdapter(sdk_config_path(name))


def load_sdk_adapter(sdk: str) -> SdkAdapter:
    return get_sdk_adapter(sdk)


def clear_sdk_adapter_cache() -> None:
    get_sdk_adapter.cache_clear()
