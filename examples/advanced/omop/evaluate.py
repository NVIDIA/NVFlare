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

import duckdb
import numpy as np
from client import newton_raphson_terms, predict
from cohort import CELECOXIB, CONDITIONS, DRUGS, FEATURES, MALE, is_test, load_cohort
from prepare_data import DATA_ROOT
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

MODEL = "/tmp/nvflare/simulation/omop_gi_bleed/server/simulate_job/models/weights.npy"


def fit(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fit an unpenalized logistic regression and return the intercept and coefficients as one column."""
    model = LogisticRegression(C=np.inf, solver="newton-cholesky").fit(features, labels.ravel())
    return np.concatenate((model.intercept_, model.coef_[0])).reshape(-1, 1)


def odds_ratios(theta: np.ndarray, hessian: np.ndarray) -> list[str]:
    """Format every covariate's odds ratio with its 95% Wald confidence interval."""
    beta = theta[1:, 0]
    margin = 1.96 * np.sqrt(np.diag(np.linalg.inv(hessian)))[1:]
    return [f"{np.exp(b):.2f} ({np.exp(b - m):.2f}-{np.exp(b + m):.2f})" for b, m in zip(beta, margin)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=DATA_ROOT)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    databases = sorted(Path(args.data_root).glob("site-*.duckdb"))
    cohorts = {}
    for database in databases:
        person_ids, features, labels = load_cohort(database)
        test = is_test(person_ids)
        cohorts[database.stem] = (features[~test], labels[~test], features[test], labels[test])
    federated = np.load(args.model).astype(np.float64)
    pooled = fit(np.vstack([c[0] for c in cohorts.values()]), np.vstack([c[1] for c in cohorts.values()]))
    print(f"Largest difference between federated and pooled coefficients: {np.abs(federated - pooled).max():.1e}")

    held_out = {site: (c[2], c[3]) for site, c in cohorts.items()}
    held_out["all sites"] = (np.vstack([c[2] for c in cohorts.values()]), np.vstack([c[3] for c in cohorts.values()]))
    aurocs = {
        site: roc_auc_score(labels, predict(federated, features)) for site, (features, labels) in held_out.items()
    }
    print("Held-out AUROC of the federated model:", ", ".join(f"{site} {auroc:.3f}" for site, auroc in aurocs.items()))

    hessian = sum(newton_raphson_terms(federated, c[0], c[1])["hessian"] for c in cohorts.values())
    estimates = {"federated": odds_ratios(federated, hessian)}
    for site, (features, labels, _, _) in cohorts.items():
        local = fit(features, labels)
        estimates[f"{site} alone"] = odds_ratios(local, newton_raphson_terms(local, features, labels)["hessian"])

    concepts = {"male": MALE, "celecoxib": CELECOXIB, **CONDITIONS, **DRUGS}
    with duckdb.connect(str(databases[0]), read_only=True) as con:
        names = dict(
            con.execute(
                "select concept_id, concept_name from concept where concept_id in (select unnest(?::bigint[]))",
                [list(concepts.values())],
            ).fetchall()
        )
    print(f"\n{'Odds ratio (95% CI)':<36}" + "".join(f"{name:>20}" for name in estimates))
    for row, feature in enumerate(FEATURES):
        concept_id = concepts.get(feature)
        covariate = f"{names[concept_id]} ({concept_id})" if concept_id else "Age, per 10 years"
        print(f"{covariate:<36}" + "".join(f"{estimate[row]:>20}" for estimate in estimates.values()))


if __name__ == "__main__":
    main()
