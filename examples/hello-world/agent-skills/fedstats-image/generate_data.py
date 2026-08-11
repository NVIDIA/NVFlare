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

"""Generate deterministic, synthetic per-site PNG inputs for this example."""

from pathlib import Path

import numpy as np
from PIL import Image


def main():
    generator = np.random.default_rng(19)
    for site, mean in {"site-1": 80, "site-2": 165}.items():
        directory = Path("data") / site
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(110):
            pixels = np.clip(generator.normal(mean, 25, (8, 8)), 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(directory / f"image-{index:03d}.png")
    (Path("data") / "site-1" / "corrupt.png").write_bytes(b"not a PNG")


if __name__ == "__main__":
    main()
