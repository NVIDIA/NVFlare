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

"""Deterministic synthetic patients in the shape of the OMOP Common Data Model.

No real patient is involved. Tables carry the CDM column names, types and keys, and the rows follow simple clinical
patterns (diabetics have HbA1c results and metformin, hypertensive patients have blood-pressure readings and
lisinopril). Concept ids are a small hand-picked set; they are not validated against the OHDSI vocabulary, and the data
is not an OMOP-conformant dataset. The catalog publishes only the schema, never these rows.

This generator belongs to the service-backed example. The simulation example uses its own smaller fixtures.
"""

import random
from datetime import date, datetime, timedelta

# One line per column: "name type", with a trailing "*" for NOT NULL. Types: int, date, datetime, str, float.
_PERSON = """person_id int*, gender_concept_id int*, year_of_birth int*, month_of_birth int, day_of_birth int,
    birth_datetime datetime, race_concept_id int*, ethnicity_concept_id int*, location_id int, provider_id int,
    care_site_id int, person_source_value str, gender_source_value str, gender_source_concept_id int,
    race_source_value str, race_source_concept_id int, ethnicity_source_value str, ethnicity_source_concept_id int"""
_OBSERVATION_PERIOD = """observation_period_id int*, person_id int*, observation_period_start_date date*,
    observation_period_end_date date*, period_type_concept_id int*"""
_VISIT_54 = """visit_occurrence_id int*, person_id int*, visit_concept_id int*, visit_start_date date*,
    visit_start_datetime datetime, visit_end_date date*, visit_end_datetime datetime, visit_type_concept_id int*,
    provider_id int, care_site_id int, visit_source_value str, visit_source_concept_id int,
    admitted_from_concept_id int, admitted_from_source_value str, discharged_to_concept_id int,
    discharged_to_source_value str, preceding_visit_occurrence_id int"""
_VISIT_53 = """visit_occurrence_id int*, person_id int*, visit_concept_id int*, visit_start_date date*,
    visit_start_datetime datetime, visit_end_date date*, visit_end_datetime datetime, visit_type_concept_id int*,
    provider_id int, care_site_id int, visit_source_value str, visit_source_concept_id int,
    admitting_source_concept_id int, admitting_source_value str, discharge_to_concept_id int,
    discharge_to_source_value str, preceding_visit_occurrence_id int"""
_CONDITION = """condition_occurrence_id int*, person_id int*, condition_concept_id int*, condition_start_date date*,
    condition_start_datetime datetime, condition_end_date date, condition_end_datetime datetime,
    condition_type_concept_id int*, condition_status_concept_id int, stop_reason str, provider_id int,
    visit_occurrence_id int, visit_detail_id int, condition_source_value str, condition_source_concept_id int,
    condition_status_source_value str"""
_DRUG = """drug_exposure_id int*, person_id int*, drug_concept_id int*, drug_exposure_start_date date*,
    drug_exposure_start_datetime datetime, drug_exposure_end_date date*, drug_exposure_end_datetime datetime,
    verbatim_end_date date, drug_type_concept_id int*, stop_reason str, refills int, quantity float, days_supply int,
    sig str, route_concept_id int, lot_number str, provider_id int, visit_occurrence_id int, visit_detail_id int,
    drug_source_value str, drug_source_concept_id int, route_source_value str, dose_unit_source_value str"""
_MEASUREMENT = """measurement_id int*, person_id int*, measurement_concept_id int*, measurement_date date*,
    measurement_datetime datetime, measurement_time str, measurement_type_concept_id int*, operator_concept_id int,
    value_as_number float, value_as_concept_id int, unit_concept_id int, range_low float, range_high float,
    provider_id int, visit_occurrence_id int, visit_detail_id int, measurement_source_value str,
    measurement_source_concept_id int, unit_source_value str, unit_source_concept_id int, value_source_value str,
    measurement_event_id int, meas_event_field_concept_id int"""


def _columns(spec):
    result = []
    for item in spec.replace("\n", " ").split(","):
        name, kind = item.split()
        result.append((name, kind.rstrip("*"), kind.endswith("*")))
    return result


def table_columns(version):
    """{table: [(column, type, required)]} for an OMOP CDM version, "5.4" or "5.3"."""
    return {
        "person": _columns(_PERSON),
        "observation_period": _columns(_OBSERVATION_PERIOD),
        "visit_occurrence": _columns(_VISIT_54 if version == "5.4" else _VISIT_53),
        "condition_occurrence": _columns(_CONDITION),
        "drug_exposure": _columns(_DRUG),
        "measurement": _columns(_MEASUREMENT),
    }


PRIMARY_KEYS = {
    "person": ["person_id"],
    "observation_period": ["observation_period_id"],
    "visit_occurrence": ["visit_occurrence_id"],
    "condition_occurrence": ["condition_occurrence_id"],
    "drug_exposure": ["drug_exposure_id"],
    "measurement": ["measurement_id"],
}
# (table, column, referenced table, referenced column)
FOREIGN_KEYS = [
    (table, column, target, column if column == "person_id" else f"{target}_id")
    for table in ("observation_period", "visit_occurrence", "condition_occurrence", "drug_exposure", "measurement")
    for column, target in (("person_id", "person"), ("visit_occurrence_id", "visit_occurrence"))
    if not (table == "observation_period" and column == "visit_occurrence_id")
    and not (table == "visit_occurrence" and column == "visit_occurrence_id")
]

# What each site holds: its CDM version and which tables it has, so the three sites differ the way real ones do.
SITE_OMOP = {
    "site-1": (
        "5.4",
        ["person", "observation_period", "visit_occurrence", "condition_occurrence", "drug_exposure", "measurement"],
        150,
    ),
    "site-2": (
        "5.4",
        ["person", "observation_period", "visit_occurrence", "condition_occurrence", "drug_exposure"],
        200,
    ),
    "site-3": ("5.3", ["person", "observation_period", "visit_occurrence", "condition_occurrence", "measurement"], 120),
}

# concept_id, ICD-10-CM source value, base prevalence, extra prevalence for each year of age over 30
_CONDITIONS = [
    (201826, "E11.9", 0.04, 0.0035),  # type 2 diabetes mellitus
    (320128, "I10", 0.08, 0.0080),  # essential hypertension
    (316139, "I50.9", 0.00, 0.0020),  # heart failure
    (255573, "J44.9", 0.00, 0.0025),  # chronic obstructive lung disease
    (317009, "J45.909", 0.06, 0.0),  # asthma
    (440383, "F32.9", 0.05, 0.0010),  # depressive disorder
    (4329847, "I21.9", 0.00, 0.0012),  # myocardial infarction
]
# condition concept -> (drug ingredient concept, drug source value, quantity, days supply)
_TREATMENT = {
    201826: [(1503297, "metformin 500 MG Oral Tablet", 60, 30)],
    320128: [(1308216, "lisinopril 10 MG Oral Tablet", 30, 30), (1307046, "metoprolol 50 MG Oral Tablet", 60, 30)],
    316139: [(1307046, "metoprolol 50 MG Oral Tablet", 60, 30)],
    4329847: [(1112807, "aspirin 81 MG Oral Tablet", 30, 30), (1545958, "atorvastatin 40 MG Oral Tablet", 30, 30)],
}
_EHR = 32817  # type concept: EHR
_ORAL = 4132161  # route concept: oral
_VISITS = [(9202, "outpatient", 0.82), (9203, "emergency", 0.10), (9201, "inpatient", 0.08)]
_RACES = [(8527, "White", 0.62), (8516, "Black or African American", 0.18), (8515, "Asian", 0.12), (0, "Other", 0.08)]
_ETHNICITIES = [(38003564, "Not Hispanic or Latino", 0.80), (38003563, "Hispanic or Latino", 0.20)]
# concept, LOINC code, unit concept, unit text, range low, range high
_BP_SYSTOLIC = (3004249, "8480-6", 8876, "mm[Hg]", 90.0, 120.0)
_BP_DIASTOLIC = (3012888, "8462-4", 8876, "mm[Hg]", 60.0, 80.0)
_WEIGHT = (3025315, "29463-7", 9529, "kg", None, None)
_BMI = (3038553, "39156-5", 9531, "kg/m2", 18.5, 24.9)
_HBA1C = (3004410, "4548-4", 8554, "%", 4.0, 5.6)
_CREATININE = (3016723, "2160-0", 8840, "mg/dL", 0.6, 1.3)


def _pick(rng, options):
    return rng.choices(options, weights=[o[-1] for o in options])[0]


def _at(day, rng):
    return datetime.combine(day, datetime.min.time()) + timedelta(hours=rng.randint(7, 18), minutes=rng.randint(0, 59))


def generate(site):
    """{table: {column: [values]}} for the site's tables, as columns in the CDM order."""
    version, tables, people = SITE_OMOP[site]
    rng = random.Random(sum(map(ord, site)) * 7919)
    base = int(site.rsplit("-", 1)[1]) * 1_000_000
    rows = {name: [] for name in table_columns(version)}
    ids = {name: base for name in rows}

    def add(table, **values):
        ids[table] += 1
        rows[table].append(values)
        return ids[table]

    five_three = version == "5.3"
    for _ in range(people):
        age = rng.randint(18, 90)
        born = date(2024, 6, 30) - timedelta(days=age * 365 + rng.randint(0, 364))
        female = rng.random() < 0.52
        race, race_text, _w = _pick(rng, _RACES)
        ethnicity, ethnicity_text, _w = _pick(rng, _ETHNICITIES)
        person = ids["person"] + 1
        add(
            "person",
            person_id=person,
            gender_concept_id=8532 if female else 8507,
            year_of_birth=born.year,
            month_of_birth=born.month,
            day_of_birth=born.day,
            birth_datetime=datetime.combine(born, datetime.min.time()),
            race_concept_id=race,
            ethnicity_concept_id=ethnicity,
            person_source_value=f"MRN-{person}",
            gender_source_value="F" if female else "M",
            gender_source_concept_id=0,
            race_source_value=race_text,
            race_source_concept_id=0,
            ethnicity_source_value=ethnicity_text,
            ethnicity_source_concept_id=0,
        )
        start = date(2019, 1, 1) + timedelta(days=rng.randint(0, 700))
        end = date(2024, 12, 31)
        add(
            "observation_period",
            observation_period_id=ids["observation_period"] + 1,
            person_id=person,
            observation_period_start_date=start,
            observation_period_end_date=end,
            period_type_concept_id=_EHR,
        )
        conditions = [c for c in _CONDITIONS if rng.random() < c[2] + c[3] * max(age - 30, 0)]
        diagnosed = {c[0] for c in conditions}
        weight = rng.gauss(88 if 201826 in diagnosed else 76, 13)
        height_m = rng.gauss(1.62 if female else 1.76, 0.07)
        visits = []
        for _ in range(rng.randint(2, 6)):
            concept, text, _w = _pick(rng, _VISITS)
            day = start + timedelta(days=rng.randint(0, (end - start).days - 10))
            stay = rng.randint(2, 7) if concept == 9201 else 0
            visit = ids["visit_occurrence"] + 1
            extra = (
                {"admitting_source_concept_id": 0, "discharge_to_concept_id": 0}
                if five_three
                else {"admitted_from_concept_id": 0, "discharged_to_concept_id": 0}
            )
            add(
                "visit_occurrence",
                visit_occurrence_id=visit,
                person_id=person,
                visit_concept_id=concept,
                visit_start_date=day,
                visit_start_datetime=_at(day, rng),
                visit_end_date=day + timedelta(days=stay),
                visit_end_datetime=_at(day + timedelta(days=stay), rng),
                visit_type_concept_id=_EHR,
                visit_source_value=text,
                visit_source_concept_id=0,
                **extra,
            )
            visits.append((visit, day))
            systolic = rng.gauss(142 if 320128 in diagnosed else 122, 10)
            for (concept_id, loinc, unit, unit_text, low, high), value in (
                (_BP_SYSTOLIC, systolic),
                (_BP_DIASTOLIC, systolic * 0.6 + rng.gauss(0, 4)),
                (_WEIGHT, weight + rng.gauss(0, 1.5)),
                (_BMI, weight / height_m**2),
            ):
                add(
                    "measurement",
                    measurement_id=ids["measurement"] + 1,
                    person_id=person,
                    measurement_concept_id=concept_id,
                    measurement_date=day,
                    measurement_datetime=_at(day, rng),
                    measurement_type_concept_id=_EHR,
                    value_as_number=round(value, 1),
                    unit_concept_id=unit,
                    range_low=low,
                    range_high=high,
                    visit_occurrence_id=visit,
                    measurement_source_value=loinc,
                    measurement_source_concept_id=0,
                    unit_source_value=unit_text,
                    unit_source_concept_id=0,
                )
            if 201826 in diagnosed and rng.random() < 0.6:
                a1c = _HBA1C
                add(
                    "measurement",
                    measurement_id=ids["measurement"] + 1,
                    person_id=person,
                    measurement_concept_id=a1c[0],
                    measurement_date=day,
                    measurement_datetime=_at(day, rng),
                    measurement_type_concept_id=_EHR,
                    value_as_number=round(rng.gauss(7.8, 1.2), 1),
                    unit_concept_id=a1c[2],
                    range_low=a1c[4],
                    range_high=a1c[5],
                    visit_occurrence_id=visit,
                    measurement_source_value=a1c[1],
                    measurement_source_concept_id=0,
                    unit_source_value=a1c[3],
                    unit_source_concept_id=0,
                )
            if (316139 in diagnosed or 201826 in diagnosed) and rng.random() < 0.4:
                cr = _CREATININE
                add(
                    "measurement",
                    measurement_id=ids["measurement"] + 1,
                    person_id=person,
                    measurement_concept_id=cr[0],
                    measurement_date=day,
                    measurement_datetime=_at(day, rng),
                    measurement_type_concept_id=_EHR,
                    value_as_number=round(rng.gauss(1.1, 0.3), 2),
                    unit_concept_id=cr[2],
                    range_low=cr[4],
                    range_high=cr[5],
                    visit_occurrence_id=visit,
                    measurement_source_value=cr[1],
                    measurement_source_concept_id=0,
                    unit_source_value=cr[3],
                    unit_source_concept_id=0,
                )
        for concept, icd, _base, _slope in conditions:
            visit, day = rng.choice(visits)
            add(
                "condition_occurrence",
                condition_occurrence_id=ids["condition_occurrence"] + 1,
                person_id=person,
                condition_concept_id=concept,
                condition_start_date=day,
                condition_start_datetime=_at(day, rng),
                condition_type_concept_id=_EHR,
                condition_status_concept_id=0,
                visit_occurrence_id=visit,
                condition_source_value=icd,
                condition_source_concept_id=0,
            )
            for drug, text, quantity, days_supply in _TREATMENT.get(concept, []):
                for refill in range(rng.randint(1, 4)):
                    fill = day + timedelta(days=refill * days_supply)
                    if fill >= end:
                        break
                    add(
                        "drug_exposure",
                        drug_exposure_id=ids["drug_exposure"] + 1,
                        person_id=person,
                        drug_concept_id=drug,
                        drug_exposure_start_date=fill,
                        drug_exposure_start_datetime=_at(fill, rng),
                        drug_exposure_end_date=fill + timedelta(days=days_supply),
                        drug_type_concept_id=_EHR,
                        refills=3 - refill,
                        quantity=float(quantity),
                        days_supply=days_supply,
                        route_concept_id=_ORAL,
                        visit_occurrence_id=visit,
                        drug_source_value=text,
                        drug_source_concept_id=0,
                        route_source_value="PO",
                    )
    columns = table_columns(version)
    return {
        name: {column: [row.get(column) for row in rows[name]] for column, _kind, _required in columns[name]}
        for name in tables
    }


def schema(site):
    """{table: [(column, type, required)]} for the tables the site holds."""
    version, tables, _people = SITE_OMOP[site]
    return {name: columns for name, columns in table_columns(version).items() if name in tables}


def table_keys(site):
    """The primary and foreign keys of each table this site has, for the database tables it creates."""
    _version, tables, _people = SITE_OMOP[site]
    return {
        table: {
            "primary_key": PRIMARY_KEYS[table],
            "relationships": [
                {"columns": [column], "target_table": target, "target_columns": [referenced]}
                for t, column, target, referenced in FOREIGN_KEYS
                if t == table and target in tables and column in {c for c, _k, _r in schema(site)[table]}
            ],
        }
        for table in tables
    }


_SQL_TYPES = {"int": "INTEGER", "date": "DATE", "datetime": "TIMESTAMP", "str": "VARCHAR(50)", "float": "FLOAT"}


def create_statements(site, schema_prefix=""):
    """CREATE TABLE statements, with primary and foreign keys, in an order that satisfies the foreign keys.

    schema_prefix, for example "omop.", qualifies the table names for a database that has schemas."""
    declared = table_keys(site)
    statements = []
    for table in sorted(declared, key=lambda t: (t != "person", t != "visit_occurrence", t)):
        parts = [
            f"{name} {_SQL_TYPES[kind]}{' NOT NULL' if required else ''}"
            for name, kind, required in schema(site)[table]
        ]
        parts.append(f"PRIMARY KEY ({', '.join(declared[table]['primary_key'])})")
        for relation in declared[table]["relationships"]:
            parts.append(
                f"FOREIGN KEY ({', '.join(relation['columns'])}) REFERENCES "
                f"{schema_prefix}{relation['target_table']} ({', '.join(relation['target_columns'])})"
            )
        statements.append((table, f"CREATE TABLE {schema_prefix}{table} ({', '.join(parts)})"))
    return statements


def insert_rows(data, table, as_text=False):
    """The rows of one table as tuples. as_text turns dates and times into ISO strings, for SQLite."""

    def value(v):
        if as_text and isinstance(v, datetime):
            return v.isoformat(sep=" ")
        return v.isoformat() if as_text and isinstance(v, date) else v

    return [tuple(value(v) for v in row) for row in zip(*data[table].values())]
