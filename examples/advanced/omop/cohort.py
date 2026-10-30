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

"""Study definition that every site runs unchanged against its own OMOP CDM.

Target cohort: patients at their first celecoxib or diclofenac exposure, observed for a year before and after it,
without gastrointestinal hemorrhage on or before it.
Outcome: gastrointestinal hemorrhage within a year after that first exposure.
Covariates: age, sex, whether celecoxib was started, and conditions and drugs recorded on or before the first exposure.
"""

from collections.abc import Mapping
from pathlib import Path

import duckdb
import numpy as np

CELECOXIB = 1118084
DICLOFENAC = 1124300
GI_HEMORRHAGE = 192671
MALE = 8507
CONDITIONS: Mapping[str, int] = {
    "peptic_ulcer": 4027663,
    "esophagitis": 30753,
    "angiodysplasia_of_stomach": 4310024,
    "diverticular_disease": 4266809,
    "ulcerative_colitis": 81893,
    "polyp_of_colon": 4285898,
}
DRUGS: Mapping[str, int] = {
    "aspirin": 1112807,
    "ibuprofen": 1177480,
    "naproxen": 1115008,
}
FEATURES = ("age_decades", "male", "celecoxib", *CONDITIONS, *DRUGS)
HISTORY = ", ".join(str(concept_id) for concept_id in (*CONDITIONS.values(), *DRUGS.values()))
FLAGS = ",\n    ".join(
    f"count(h.concept_id) filter (where h.concept_id = {concept_id}) > 0 as {name}"
    for name, concept_id in {**CONDITIONS, **DRUGS}.items()
)

COHORT_SQL = f"""
with gi_hemorrhage as (
    select co.person_id, co.condition_start_date
    from condition_occurrence co
    join concept_ancestor ca on ca.descendant_concept_id = co.condition_concept_id
    where ca.ancestor_concept_id = {GI_HEMORRHAGE}
),
first_exposure as (
    select de.person_id,
           min(de.drug_exposure_start_date) as index_date,
           min(de.drug_exposure_start_date) filter (where ca.ancestor_concept_id = {CELECOXIB}) as first_celecoxib
    from drug_exposure de
    join concept_ancestor ca on ca.descendant_concept_id = de.drug_concept_id
    where ca.ancestor_concept_id in ({CELECOXIB}, {DICLOFENAC})
    group by de.person_id
),
target as (
    select f.*
    from first_exposure f
    join observation_period op on op.person_id = f.person_id
    where f.index_date - interval 365 day >= op.observation_period_start_date
      and f.index_date + interval 365 day <= op.observation_period_end_date
      and not exists (
          select 1 from gi_hemorrhage g where g.person_id = f.person_id and g.condition_start_date <= f.index_date
      )
),
history as (
    select t.person_id, ca.ancestor_concept_id as concept_id
    from target t
    join condition_occurrence co on co.person_id = t.person_id and co.condition_start_date <= t.index_date
    join concept_ancestor ca on ca.descendant_concept_id = co.condition_concept_id
    where ca.ancestor_concept_id in ({HISTORY})
    union
    select t.person_id, ca.ancestor_concept_id
    from target t
    join drug_exposure de on de.person_id = t.person_id and de.drug_exposure_start_date <= t.index_date
    join concept_ancestor ca on ca.descendant_concept_id = de.drug_concept_id
    where ca.ancestor_concept_id in ({HISTORY})
),
outcome as (
    select distinct t.person_id
    from target t
    join gi_hemorrhage g on g.person_id = t.person_id
    where g.condition_start_date > t.index_date
      and g.condition_start_date <= t.index_date + interval 365 day
)
select
    t.person_id,
    (year(t.index_date) - p.year_of_birth - 40) / 10.0 as age_decades,
    p.gender_concept_id = {MALE} as male,
    coalesce(t.first_celecoxib = t.index_date, false) as celecoxib,
    {FLAGS},
    t.person_id in (from outcome) as outcome
from target t
join person p on p.person_id = t.person_id
left join history h on h.person_id = t.person_id
group by all
order by t.person_id
"""


def load_cohort(database: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the study query against one site's OMOP CDM.

    Args:
        database: The site's DuckDB database.

    Returns:
        Person ids, the covariate matrix with columns ``FEATURES``, and the outcome as a column vector.
    """
    with duckdb.connect(str(database), read_only=True) as con:
        rows = con.execute(COHORT_SQL).fetchnumpy()
    features = np.column_stack([rows[name] for name in FEATURES]).astype(np.float64)
    return rows["person_id"], features, rows["outcome"].astype(np.float64).reshape(-1, 1)
