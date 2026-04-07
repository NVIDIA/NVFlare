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

"""Generic dataset generation utilities.

Provides a topological sort over an attribute dependency graph and a
``generate`` function that walks the sorted graph to populate a
``pd.DataFrame`` level-by-level.

Typical usage::

    from data_generation.attributes import (
        get_per_participant_attributes,
        get_payment_core_attributes,
        get_payment_amount_attributes,
    )
    from data_generation.dataset import topological_sort, generate

    graph = get_per_participant_attributes(("DEBITOR", "CREDITOR"))
    get_payment_core_attributes(("DEBITOR", "CREDITOR"), graph)
    get_payment_amount_attributes(("DEBITOR", "CREDITOR"), graph)

    providers = { attr: ... for attr in graph }   # map each attr → provider
    df = generate(graph, providers, n_rows=1000)
"""

from collections import deque
from collections.abc import Mapping

import pandas as pd
from data_generation.dataset_attribute import PaymentDatasetAttributeGroup
from data_generation.synthetic_data_provider.synthetic_data_provider import SyntheticDataProvider


def topological_sort(
    dependency_graph: Mapping[PaymentDatasetAttributeGroup, list[str]],
) -> list[PaymentDatasetAttributeGroup]:
    """Return attributes in dependency-respecting order (Kahn's algorithm).

    Args:
        dependency_graph: Maps each attribute (or attribute group) to the
            column names it depends on.  Attributes with an empty list have
            no dependencies and may appear first.

    Returns:
        A list of attributes ordered so that every attribute appears after
        all attributes whose columns it depends on.

    Raises:
        ValueError: If the graph contains a cycle.
    """
    # Reverse index: column name → attribute that produces it
    column_producer: dict[str, PaymentDatasetAttributeGroup] = {}
    for attr in dependency_graph:
        for col_name in attr.names:
            column_producer[col_name] = attr

    # Build adjacency list and in-degree count
    in_degree: dict[PaymentDatasetAttributeGroup, int] = dict.fromkeys(dependency_graph, 0)
    dependents: dict[PaymentDatasetAttributeGroup, list[PaymentDatasetAttributeGroup]] = {
        attr: [] for attr in dependency_graph
    }

    for attr, dep_columns in dependency_graph.items():
        seen_producers: set[PaymentDatasetAttributeGroup] = set()
        for col_name in dep_columns:
            producer = column_producer.get(col_name)
            if producer is not None and producer is not attr and producer not in seen_producers:
                in_degree[attr] += 1
                dependents[producer].append(attr)
                seen_producers.add(producer)

    # Kahn's BFS
    queue: deque[PaymentDatasetAttributeGroup] = deque(attr for attr, deg in in_degree.items() if deg == 0)
    ordered: list[PaymentDatasetAttributeGroup] = []

    while queue:
        attr = queue.popleft()
        ordered.append(attr)
        for dep in dependents[attr]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    if len(ordered) != len(dependency_graph):
        raise ValueError(f"Dependency cycle detected: sorted {len(ordered)} of " f"{len(dependency_graph)} attributes")

    return ordered


def generate(
    dependency_graph: Mapping[PaymentDatasetAttributeGroup, list[str]],
    providers: Mapping[PaymentDatasetAttributeGroup, SyntheticDataProvider],
    n_rows: int,
    **kwargs,
) -> pd.DataFrame:
    """Generate a complete DataFrame by walking the dependency graph.

    Topologically sorts the graph, then iterates through attributes in
    dependency order — calling each attribute's ``emit()`` with the
    appropriate provider and dependent columns.

    Args:
        dependency_graph: Maps each attribute to its dependent column names.
        providers: Maps each attribute to the ``SyntheticDataProvider``
            instance it should use for generation.
        n_rows: Number of rows to generate.
        **kwargs: Extra arguments forwarded to every attribute's ``emit()``
            call (e.g. ``country_static_data``, sampling configs).

    Returns:
        A fully populated ``pd.DataFrame`` with one column per attribute name.

    Raises:
        KeyError: If an attribute is missing from the *providers* mapping.
        ValueError: If the dependency graph contains a cycle.
    """
    ordered = topological_sort(dependency_graph)
    df = pd.DataFrame(index=range(n_rows))

    for attr in ordered:
        dep_columns = dependency_graph[attr] or None
        provider = providers[attr]
        result = attr.emit(df, provider, dep_columns, **kwargs)

        if isinstance(result, pd.DataFrame):
            for result_col, attr_name in zip(result.columns, attr.names):
                df[attr_name] = result[result_col]
        else:
            df[attr.names[0]] = result

    return df
