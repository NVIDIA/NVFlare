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

from pathlib import Path

import duckdb
import numpy as np
import pytest
from cohort import FEATURES, load_cohort

CDM_SQL = """
create table person (person_id bigint, gender_concept_id bigint, year_of_birth bigint);
create table observation_period (
    person_id bigint, observation_period_start_date date, observation_period_end_date date
);
create table drug_exposure (person_id bigint, drug_concept_id bigint, drug_exposure_start_date date);
create table condition_occurrence (person_id bigint, condition_concept_id bigint, condition_start_date date);
create table concept_ancestor (ancestor_concept_id bigint, descendant_concept_id bigint);

insert into concept_ancestor values
    (1118084, 1118088), (1124300, 40162359), (1112807, 19059056), (4027663, 4027663), (192671, 192671), (192671, 1);
insert into person values
    (1, 8507, 1960), (2, 8532, 1970), (3, 8532, 1960), (4, 8507, 1960), (5, 8507, 1960), (6, 8507, 1960),
    (7, 8532, 1965);
insert into observation_period select person_id, '2000-01-01', '2010-12-31' from person;
insert into drug_exposure values
    (1, 19059056, '2004-01-01'), (1, 1118088, '2005-06-01'),
    (2, 40162359, '2006-01-01'), (2, 1118088, '2006-03-01'),
    (3, 1118088, '2000-06-01'),
    (4, 1118088, '2010-06-01'),
    (5, 19059056, '2005-06-01'),
    (6, 1118088, '2005-06-01'),
    (7, 40162359, '2005-01-01'), (7, 1118088, '2005-01-01');
insert into condition_occurrence values
    (1, 4027663, '2003-01-01'), (1, 1, '2005-12-01'),
    (2, 4027663, '2006-06-01'), (2, 192671, '2007-03-01'),
    (5, 192671, '2005-12-01'),
    (6, 192671, '2004-01-01'), (6, 192671, '2005-12-01');
"""


@pytest.fixture
def cdm(tmp_path: Path) -> Path:
    database = tmp_path / "site-1.duckdb"
    with duckdb.connect(str(database)) as con:
        con.execute(CDM_SQL)
    return database


def test_cohort_applies_washout_follow_up_prior_outcome_and_concept_hierarchy(cdm: Path) -> None:
    person_ids, features, labels = load_cohort(cdm)

    expected = np.zeros((3, len(FEATURES)))
    expected[:, FEATURES.index("age_decades")] = [0.5, -0.4, 0.0]
    expected[0, [FEATURES.index(name) for name in ("male", "celecoxib", "peptic_ulcer", "aspirin")]] = 1
    expected[2, FEATURES.index("celecoxib")] = 1
    np.testing.assert_array_equal(person_ids, [1, 2, 7])
    np.testing.assert_allclose(features, expected)
    np.testing.assert_array_equal(labels, [[1.0], [0.0], [0.0]])
