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

"""Site-side Newton-Raphson logistic regression on the study cohort extracted from the site's OMOP CDM."""

import argparse
from pathlib import Path

import numpy as np
from cohort import load_cohort

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.np.constants import NPConstants


def newton_raphson_terms(theta: np.ndarray, features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Sum the log-likelihood gradient and the negative Hessian over the local patients."""
    design = np.column_stack((np.ones(len(features)), features))
    proba = 1.0 / (1.0 + np.exp(-(design @ theta)))
    return {
        "gradient": design.T @ (labels - proba),
        "hessian": (design * proba * (1.0 - proba)).T @ design,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    args = parser.parse_args()

    flare.init()
    site = flare.get_site_name()
    _, features, labels = load_cohort(Path(args.data_root) / f"{site}.duckdb")
    print(f"{site}: {len(labels)} patients in the cohort, {int(labels.sum())} with GI hemorrhage")

    while flare.is_running():
        theta = flare.receive().params[NPConstants.NUMPY_KEY]
        # Unweighted sums: the server's average of them gives the Newton-Raphson step of the pooled data.
        flare.send(FLModel(params=newton_raphson_terms(theta, features, labels), params_type=ParamsType.FULL))


if __name__ == "__main__":
    main()
