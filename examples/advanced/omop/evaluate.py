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

"""Check the federated model against a pooled fit and compare its odds ratios with each site's own estimates."""

import argparse
from pathlib import Path

import numpy as np
from client import newton_raphson_terms
from cohort import FEATURES, load_cohort
from prepare_data import DATA_ROOT
from sklearn.linear_model import LogisticRegression

MODEL = "/tmp/nvflare/simulation/omop_gi_bleed/server/simulate_job/models/weights.npy"


def fit(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fit an unpenalized logistic regression and return the intercept and coefficients as one column."""
    model = LogisticRegression(C=np.inf, solver="newton-cholesky").fit(features, labels.ravel())
    return np.concatenate((model.intercept_, model.coef_[0])).reshape(-1, 1)


def odds_ratios(theta: np.ndarray, features: np.ndarray, labels: np.ndarray) -> list[str]:
    """Format every covariate's odds ratio with its 95% Wald confidence interval."""
    hessian = newton_raphson_terms(theta, features, labels)["hessian"]
    margins = 1.96 * np.sqrt(np.diag(np.linalg.inv(hessian)))[1:]
    return [f"{np.exp(b):.2f} ({np.exp(b - m):.2f}-{np.exp(b + m):.2f})" for b, m in zip(theta[1:, 0], margins)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    sites = {path.stem: load_cohort(path)[1:] for path in sorted(Path(args.data_root).glob("site-*.duckdb"))}
    features = np.vstack([x for x, _ in sites.values()])
    labels = np.vstack([y for _, y in sites.values()])
    federated = np.load(args.model).astype(np.float64)
    pooled = fit(features, labels)
    print(f"Largest difference between federated and pooled coefficients: {np.abs(federated - pooled).max():.1e}\n")

    estimates = {"federated": odds_ratios(federated, features, labels)}
    estimates.update({f"{site} alone": odds_ratios(fit(x, y), x, y) for site, (x, y) in sites.items()})
    print(f"{'Odds ratio (95% CI)':<26}" + "".join(f"{name:>20}" for name in estimates))
    for row, feature in enumerate(FEATURES):
        print(f"{feature:<26}" + "".join(f"{estimate[row]:>20}" for estimate in estimates.values()))


if __name__ == "__main__":
    main()
