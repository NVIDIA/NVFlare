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

"""Deterministic checks for conversion skill device-selection behavior."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

DEVICE_SELECTION_BEHAVIOR_ID = "device-selection-respects-availability"

DeviceSelectionStatus = Literal["pass", "fail", "missing", "not_applicable"]

_CUDA_AVAILABLE_RE = re.compile(r"torch\s*\.\s*cuda\s*\.\s*is_available\s*\(", re.IGNORECASE)
_GPU_TOKEN_RE = re.compile(r"\b(?:cuda(?::\d+)?|gpu)\b", re.IGNORECASE)
_CPU_FALLBACK_RE = re.compile(r"\b(?:cpu|none)\b", re.IGNORECASE)
_HARD_CPU_RE = re.compile(
    r"""
    (?:
        torch\s*\.\s*device\s*\(\s*["']cpu["']\s*\)
        |
        \b(?:device|DEVICE|accelerator)\s*=\s*["']cpu["']
        |
        \b(?:device|DEVICE)\s*=\s*torch\s*\.\s*device\s*\(\s*["']cpu["']\s*\)
        |
        accelerator\s*=\s*["']cpu["']
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_HARD_GPU_RE = re.compile(
    r"""
    (?:
        torch\s*\.\s*device\s*\(\s*["']cuda(?::\d+)?["']\s*\)
        |
        \b(?:device|DEVICE)\s*=\s*["']cuda(?::\d+)?["']
        |
        \b(?:device|DEVICE)\s*=\s*torch\s*\.\s*device\s*\(\s*["']cuda(?::\d+)?["']\s*\)
        |
        accelerator\s*=\s*["']gpu["']
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_ANY_DEVICE_LOGIC_RE = re.compile(
    r"torch\s*\.\s*cuda\s*\.\s*is_available|torch\s*\.\s*device|\b(?:device|DEVICE|accelerator|devices)\b|\.to\s*\(",
    re.IGNORECASE,
)
_RUNTIME_DEVICE_RE = re.compile(
    r"\b(?:device|accelerator)\s*(?:=|:)\s*['\"]?(cuda(?::\d+)?|gpu|cpu)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DeviceSelectionResult:
    """Scored device-selection behavior for one conversion run."""

    status: DeviceSelectionStatus
    evidence: str

    def as_behavior_record(self) -> dict[str, dict[str, dict[str, str]]]:
        """Return the run-record shape expected by skill eval reports."""

        return {
            "mandatory_behavior": {
                DEVICE_SELECTION_BEHAVIOR_ID: {
                    "status": self.status,
                    "evidence": self.evidence,
                }
            }
        }


def check_device_selection(
    source_text: str,
    generated_text: str,
    *,
    runtime_log: str | None = None,
    gpu_available: bool | None = None,
) -> DeviceSelectionResult:
    """Score whether a conversion preserves GPU-when-available device selection.

    The source is the condition for applicability: CPU-only source code is not
    penalized. Runtime evidence, when collected on a GPU host, wins over static
    generated-code inspection because it proves the converted job's actual
    device choice.
    """

    if not source_uses_gpu_when_available(source_text):
        return DeviceSelectionResult(
            "not_applicable",
            "source does not select CUDA/GPU conditionally with torch.cuda.is_available(); device rule not scored",
        )

    runtime_device = _runtime_device(runtime_log)
    if gpu_available is True and runtime_device:
        if runtime_device in {"cuda", "gpu"} or runtime_device.startswith("cuda:"):
            return DeviceSelectionResult(
                "pass",
                f"runtime log selected {runtime_device!r} while GPU was available",
            )
        if runtime_device == "cpu":
            return DeviceSelectionResult(
                "fail",
                "runtime log selected 'cpu' while GPU was available and source used GPU-when-available",
            )

    if source_uses_gpu_when_available(generated_text):
        return DeviceSelectionResult(
            "pass",
            "generated code keeps torch.cuda.is_available() conditional CUDA/GPU selection with CPU fallback",
        )

    if _HARD_CPU_RE.search(generated_text):
        return DeviceSelectionResult(
            "fail",
            "source used GPU-when-available but generated code hard-codes CPU device selection",
        )

    if _HARD_GPU_RE.search(generated_text):
        return DeviceSelectionResult(
            "fail",
            "source used GPU-when-available but generated code hard-codes GPU without CPU fallback",
        )

    if not _ANY_DEVICE_LOGIC_RE.search(generated_text):
        return DeviceSelectionResult(
            "missing",
            "source used GPU-when-available but generated code has no detectable device-selection logic",
        )

    return DeviceSelectionResult(
        "missing",
        "source used GPU-when-available but generated code does not show conditional CUDA/GPU selection with CPU fallback",
    )


def source_uses_gpu_when_available(text: str) -> bool:
    """Return True when code statically shows CUDA/GPU-if-available behavior."""

    for match in _CUDA_AVAILABLE_RE.finditer(text):
        start = max(match.start() - 180, 0)
        end = min(match.end() + 180, len(text))
        window = text[start:end]
        if _GPU_TOKEN_RE.search(window) and _CPU_FALLBACK_RE.search(window):
            return True
    return False


def _runtime_device(runtime_log: str | None) -> str | None:
    if not runtime_log:
        return None
    match = _RUNTIME_DEVICE_RE.search(runtime_log)
    if not match:
        return None
    return match.group(1).lower()
