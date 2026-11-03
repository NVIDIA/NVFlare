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

"""Small CoCo release helpers shared by provisioning and packaging."""

from pathlib import Path

from nvflare.lighter.constants import ProvFileName

COCO_STARTUP_PROLOGUE = "#!/usr/bin/env bash\nexec >/dev/null 2>&1\n"
COCO_RUNTIMES = {
    ("amd_sev_snp", "nvidia_cc"): "kata-qemu-nvidia-gpu-snp",
    ("amd_sev_snp", "none"): "kata-qemu-snp",
    ("intel_tdx", "nvidia_cc"): "kata-qemu-nvidia-gpu-tdx",
    ("intel_tdx", "none"): "kata-qemu-tdx",
}


def coco_runtime_class(plan):
    """Resolve the runtime class from an already validated deployment plan."""
    try:
        return COCO_RUNTIMES[(plan.cpu_tee.value, plan.gpu_tee.value)]
    except (AttributeError, KeyError):
        raise ValueError("Unsupported CoCo CPU/GPU combination") from None


def _silence_coco_startup(ctx, participant):
    """Discard host-visible startup/process output before signing the kit."""
    kit = Path(ctx.get_ws_dir(participant))
    if (kit / ProvFileName.SIGNATURE_JSON).exists():
        raise RuntimeError("CoCo startup must be silenced before SignatureBuilder")
    startup = kit / "startup/sub_start.sh"
    text = startup.read_text()
    shebang = "#!/usr/bin/env bash\n"
    if not text.startswith(shebang):
        raise RuntimeError("CoCo requires the standard Bash startup script")
    if not text.startswith(COCO_STARTUP_PROLOGUE):
        # Keep no duplicate of the original stdout/stderr descriptors. All
        # children inherit /dev/null; guest-local NVFlare file logs remain.
        startup.write_text(COCO_STARTUP_PROLOGUE + text[len(shebang) :])
