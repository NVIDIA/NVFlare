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

"""Generate deterministic, synthetic per-site CSV inputs for this example."""

import csv
from pathlib import Path
from random import Random


def main():
    for site, seed, age_offset in (("site-1", 17, 0), ("site-2", 23, 12)):
        generator = Random(seed)
        directory = Path("data") / site
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "patients.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("age", "bmi", "score"))
            writer.writeheader()
            for _ in range(110):
                writer.writerow(
                    {
                        "age": 35 + age_offset + generator.randrange(25),
                        "bmi": round(20 + generator.random() * 14, 1),
                        "score": round(0.5 + generator.random() * 0.49, 2),
                    }
                )


if __name__ == "__main__":
    main()
