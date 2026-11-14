# Federated Data Cataloging: Metadata Schema Details

**Status:** Companion to the [federated data cataloging design](federated_data_cataloging_design.md)
**Date:** November 9, 2026

This page explains the catalog metadata model and the checks that enforce it. The format itself is normative in Section 3.4 of the main design. The code is `nvflare/app_common/fdn/catalog/common/schema.py`, which has no FLARE or I/O imports. Its error messages name fields, never values from the source.

## 1. The model

A catalog describes one dataset at one site. Code uses frozen dataclasses with tuples, so a validated catalog cannot change. The wire form is a plain dict.

| Type | Holds | Notes |
| --- | --- | --- |
| `CatalogInfo` | `dataset`, `display_name`, `description`, `owner`, `source_version`, `modalities`, `operations`, `observed_at` and the objects below | The root. Identity is `dataset` plus the authenticated Study and client, never a second id. |
| `TableInfo` | `name`, `primary_key` | Names are unique within the dataset. |
| `ColumnInfo` | `name`, `type`, `nullable` | `type` is exactly what the source reports. `nullable` is true, false or null. |
| `Relationship` | `columns`, `target_table`, `target_columns` | A foreign key. Both column lists are the same nonzero length and must exist. |
| `SemanticProfile` | `standard`, `version`, `vocabulary_version`, `mapping_version`, `declaration_ref`, `validation_ref` | Declared by the site, never inferred. Every field may be null. |
| `Provenance` | `source`, `declaration_ref`, `validation_ref` | `source` is required. |
| `Resource` | `type`, `supported_search` | A non-tabular capability. Types are unique. |
| `Lineage` | `dataset`, `source_version`, `relation` | `relation` is only `derived_from`. Entries are unique by `dataset`. |
| `ReadLimits` | `max_duration_s`, `max_calls`, `max_records` | Not part of a catalog. Site defaults are 60 s per read, 1000 calls per connector read and 10000 records across the site read. Each dataset receives the remaining time and record budgets. |

## 2. Two gates

A dict becomes an accepted catalog only by passing both gates, in order.

**Gate 1, shape: `CatalogInfo.from_wire`.** Every object must be a dict with exactly the expected keys. An extra key fails with "has unsupported fields" and a missing key with "is missing [...]". Strings, lists and `nullable` must have the right type. Gate 1 does not judge values: an empty name passes.

**Gate 2, meaning: `validate(catalog, limits)`.**

- Size: `record_count()` is within `max_records`. This is checked first. The count is the dataset, every table, column and relationship, and every resource and lineage entry.
- Text: required fields are nonempty and at most 256 characters. `description` allows 4096 and may be empty.
- Uniqueness: tables, columns within a table, primary-key columns, resource types and their search names, modalities, operations and lineage datasets.
- References: primary-key columns, relationship columns, and the target table and target columns all exist in this catalog.
- Lineage `relation` is `derived_from`.

| Defect | Gate | Error |
| --- | --- | --- |
| Extra key in `metadata` | 1 | `metadata has unsupported fields` |
| `observed_at` missing | 1 | `catalog is missing ['observed_at']` |
| `nullable: "yes"` | 1 | `metadata.tables[0].columns[0].nullable must be a boolean or null` |
| Empty `display_name` | 2 | `display_name must not be empty` |
| Duplicate column | 2 | `tables[0].columns has duplicate entries` |
| Primary key on a missing column | 2 | `tables[0].primary_key references an unknown column` |
| Relationship to a missing table | 2 | `tables[1].relationships[0] references an unknown target` |
| Column lists of different length | 2 | `tables[1].relationships[0] column lists must be nonempty and of equal length` |
| Lineage relation other than `derived_from` | 2 | `lineage[0].relation is not supported` |
| Too many records | 2 | `catalog exceeds the record budget` |

## 3. Where each gate runs

The site checks its own output. The controller does not trust that and checks every entry again before it writes the job's result.

| Where | Call | On failure |
| --- | --- | --- |
| Site executor, after the rules are applied | `validate(catalog, remaining site record budget)` | The dataset is reported `unavailable` with no partial schema. A record or time overrun makes it `skipped` instead (Section 3.6 of the main design). |
| Server controller, parsing a site's result | `CatalogInfo.from_wire(entry)` for each entry | The entry is dropped and its dataset is added to that site's `unavailable` list; the site's other entries continue. |
| Server controller, before writing the result | `validate(catalog, controller limits)` | The same. |
| User, loading a saved result | `CatalogInfo.from_wire(entry)` | `CatalogVisualization` and other readers hold saved files to the same shape. |

The controller validates each catalog with its own `max_records` of 10000. A site's limits never raise it, so a site that permits a larger catalog still has that catalog rejected by the controller.

The rules are applied before validation, so a catalog is validated after excluded tables, columns and their relationships are removed. A validation failure therefore never exposes an excluded name.

## 4. Record count

`record_count()` is what the site budgets against: each completed catalog consumes records from the whole-read budget, and the next dataset gets the remainder. Example: one dataset, two tables, five columns, one relationship and one lineage entry is 10 records. The example site-1 OMOP dataset has 6 tables and 102 columns, about 1.2% of the default 10000.
