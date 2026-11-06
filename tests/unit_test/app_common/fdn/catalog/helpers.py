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

from nvflare.app_common.fdn.catalog.common.schema import (
    CatalogInfo,
    ColumnInfo,
    Lineage,
    Provenance,
    ReadLimits,
    Relationship,
    Resource,
    SemanticProfile,
    TableInfo,
)
from nvflare.app_common.fdn.catalog.site.reader import CatalogReader

LIMITS = ReadLimits(max_duration_s=60, max_calls=100, max_records=1000)


def make_catalog(**overrides) -> CatalogInfo:
    fields = dict(
        dataset="clinical-research",
        display_name="Clinical research",
        provenance=Provenance(source="parquet"),
        tables=(
            TableInfo(
                name="person",
                columns=(ColumnInfo("person_id", "int64", False), ColumnInfo("name", "string")),
                primary_key=("person_id",),
            ),
            TableInfo(
                name="visit",
                columns=(ColumnInfo("visit_id", "int64"), ColumnInfo("person_id", "int64")),
                relationships=(Relationship(("person_id",), "person", ("person_id",)),),
            ),
        ),
        resources=(Resource("Patient", ("name",)),),
        semantic_profile=SemanticProfile(standard="OMOP", version="5.4"),
        modalities=("clinical_tabular",),
        operations=("cohort.count",),
        lineage=(Lineage("raw", "v1"),),
    )
    fields.update(overrides)
    return CatalogInfo(**fields)


def make_large_catalog(dataset="wide-catalog") -> CatalogInfo:
    """Valid metadata over the former 1 MB response cap, within the record and field limits."""
    columns = tuple(ColumnInfo(f"column-{i:04d}-" + "x" * 160, "string") for i in range(6000))
    return make_catalog(dataset=dataset, tables=(TableInfo("records", columns),), resources=(), lineage=())


class StubCatalogReader(CatalogReader):
    def read_catalog(self, dataset, mounts, limits, options=None):
        return make_catalog(dataset=dataset)
