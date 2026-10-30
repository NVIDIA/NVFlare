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
from cohort import is_test, load_cohort
from sklearn.metrics import roc_auc_score

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.np.constants import NPConstants


def predict(theta: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Return the outcome probability for every row, with ``theta[0]`` as the intercept."""
    return 1.0 / (1.0 + np.exp(-(features @ theta[1:] + theta[0])))


def newton_raphson_terms(theta: np.ndarray, features: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Sum the log-likelihood gradient and the negative Hessian over the local patients."""
    design = np.column_stack((np.ones(len(features)), features))
    proba = predict(theta, features)
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
    person_ids, features, labels = load_cohort(Path(args.data_root) / f"{site}.duckdb")
    test = is_test(person_ids)
    print(f"{site}: {len(labels)} patients in the cohort, {int(labels.sum())} with GI hemorrhage")

    while flare.is_running():
        global_model = flare.receive()
        theta = global_model.params[NPConstants.NUMPY_KEY]
        metrics = {}
        if np.unique(labels[test]).size == 2:
            metrics["auroc"] = roc_auc_score(labels[test], predict(theta, features[test]))
            print(f"{site} round {global_model.current_round}: held-out AUROC {metrics['auroc']:.3f}")
        # Unweighted sums: the server's average of them gives the Newton-Raphson step of the pooled data.
        flare.send(
            FLModel(
                params=newton_raphson_terms(theta, features[~test], labels[~test]),
                params_type=ParamsType.FULL,
                metrics=metrics,
            )
        )


if __name__ == "__main__":
    main()
