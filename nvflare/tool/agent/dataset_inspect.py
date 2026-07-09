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

"""Static dataset classification for `nvflare agent inspect`.

Classifies a data-only directory as a tabular or image dataset and emits a
metadata-only evidence block. It never emits cell values or pixel data and
follows no symlinks. Read bounds: text and image reads are capped at
``max_file_bytes`` each (worst case ``max_files x max_file_bytes``);
parquet metadata reads parse only the file footer via pyarrow — they are
not capped by ``max_file_bytes`` and are accounted in ``scan`` at file
size as the upper bound.

Dataset block contract (all keys always present per modality):

- ``modality``: ``tabular`` | ``image`` | ``mixed``. Mixed means the
  minority modality is a material share (>= 10%) of the data files; it is
  reported but not routed (``target_type`` stays ``unknown_target``).
  Below the threshold the majority wins and stray files stay visible in
  ``file_census``.
- ``layout``: ``per_site_directories`` | ``flat``.
- ``file_census``: extension -> count (bounded; see ``counts_approximate``).
- ``counts_approximate``: true when a walk limit was hit.
- ``scan``: ``files_read`` / ``bytes_read`` performed by this classification.
- ``sites``: per top-level directory (or ``.`` when flat):
  - always: ``name``, ``data_files``;
  - tabular/parquet: ``format``, ``header``, ``column_count``, ``features``,
    ``dtypes``, ``row_count``, ``row_count_approximate``, and for parquet
    ``schema_available``. ``header: present`` means "names available"
    (parquet always has names when readable); ``header: ambiguous`` means
    names must be user-supplied. ``row_count`` aggregates every tabular
    file in the site; sharded CSV sites are always approximate (per-shard
    headers are unknowable), parquet shards are exact metadata sums when
    the shards agree on schema.
    Feature names are sanitized: control characters stripped and length
    capped (``feature_names_truncated`` marks a cap hit).
  - image: ``pixel_depth`` from one sample when an optional loader exists.
- ``schema_agreement`` (tabular): compares feature names, column counts,
  AND dtype classes across sites; any drift is a ``mismatch`` with per-site
  issues (``feature_names_differ`` | ``column_count_differs`` |
  ``dtypes_differ`` | ``shards_differ``, the last when shards inside one
  site disagree — flagged per site as ``shard_schema_consistent: false``).

Walk bounds: ``max_files`` counts data files so non-data clutter cannot
hide a dataset, and ``MAX_WALK_ENTRIES`` bounds total traversal (files and
directories). This deliberately differs from the code walk in
``inspector.py``, which bounds per-file scan work; the two walks bound
different costs and are kept separate on purpose.
"""

import csv
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TABULAR_EXTENSIONS = {".csv", ".tsv", ".parquet"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".nii"}
TEXT_TABULAR_EXTENSIONS = {".csv", ".tsv"}
MAX_SAMPLE_ROWS = 200
MAX_SCHEMA_COLUMNS = 512
MAX_FEATURE_NAME_CHARS = 120
MAX_WALK_ENTRIES = 50_000
# Both modalities present: classify as mixed (unrouted) when the minority
# modality is a material share of the data files; below the threshold the
# stray files (a plot.png beside CSVs) do not flip the classification.
MIXED_MINORITY_SHARE = 0.10

HEADER_PRESENT = "present"
# Per the fed-stats contract: a first row is a header only when at least one
# column shows text over an otherwise-numeric column. Anything else (all
# numeric, or text over text columns) cannot be distinguished from data, so
# the caller must obtain declared names or an explicit header statement.
HEADER_AMBIGUOUS = "ambiguous"

# float() accepts nan/inf tokens; a column of literal "nan" strings is not
# numeric evidence for statistics.
_NON_NUMERIC_FLOAT_TOKENS = {"nan", "inf", "-inf", "+inf", "infinity", "-infinity", "+infinity"}


def inspect_dataset(root: Path, max_files: int, max_file_bytes: int) -> Optional[dict]:
    """Classify a directory as a dataset; None when it holds no data files."""
    groups, census, truncated = _collect_data_files(root, max_files)
    tabular_count = sum(census.get(ext, 0) for ext in TABULAR_EXTENSIONS)
    image_count = sum(n for ext, n in census.items() if ext in IMAGE_EXTENSIONS or ext == ".nii.gz")
    if tabular_count == 0 and image_count == 0:
        return None
    minority = min(tabular_count, image_count)
    if minority > 0 and minority / (tabular_count + image_count) >= MIXED_MINORITY_SHARE:
        modality = "mixed"
    else:
        modality = "tabular" if tabular_count > image_count else "image"

    # Dataset classification reads data bytes (bounded, metadata-only);
    # account for every read so the inspection stays auditable.
    reads = {"files_read": 0, "bytes_read": 0}
    sites = []
    for name in sorted(groups):
        files = groups[name]
        site: Dict = {"name": name, "data_files": len(files)}
        if modality == "tabular":
            site.update(_tabular_schema(files, max_file_bytes, reads))
        elif modality == "image":
            site["pixel_depth"] = _sample_pixel_depth(files, reads)
        sites.append(site)

    dataset: Dict = {
        "modality": modality,
        "layout": "per_site_directories" if any(s["name"] != "." for s in sites) else "flat",
        "file_census": {ext: census[ext] for ext in sorted(census)},
        "counts_approximate": truncated,
        "scan": reads,
        "sites": sites,
    }
    if modality == "tabular":
        dataset["schema_agreement"] = _schema_agreement(sites)
    return dataset


def _collect_data_files(root: Path, max_files: int):
    """Bounded, sorted, symlink-free walk grouping data files by top-level dir.

    ``max_files`` counts *data* files, so non-data clutter cannot exhaust the
    budget before any data file is seen; ``MAX_WALK_ENTRIES`` bounds total
    traversal work, counting directories as well as files.
    """
    groups: Dict[str, List[Path]] = {}
    census: Dict[str, int] = {}
    data_files_seen = 0
    entries_seen = 0
    truncated = False
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda p: p.name, reverse=True)
        except OSError:
            continue
        for child in children:
            if child.is_symlink():
                continue
            entries_seen += 1
            if entries_seen > MAX_WALK_ENTRIES:
                truncated = True
                stack.clear()
                break
            if child.is_dir():
                stack.append(child)
                continue
            if not child.is_file():
                continue
            ext = _data_extension(child)
            if ext is None:
                continue
            data_files_seen += 1
            if data_files_seen > max_files:
                truncated = True
                stack.clear()
                break
            census[ext] = census.get(ext, 0) + 1
            rel = child.relative_to(root)
            group = rel.parts[0] if len(rel.parts) > 1 else "."
            groups.setdefault(group, []).append(child)
    for files in groups.values():
        files.sort()
    return groups, census, truncated


def _data_extension(path: Path) -> Optional[str]:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    suffix = path.suffix.lower()
    if suffix in TABULAR_EXTENSIONS or suffix in IMAGE_EXTENSIONS:
        return suffix
    return None


def _null_schema(fmt: str) -> dict:
    """Stable site shape: every schema key present even for degenerate files."""
    return {
        "format": fmt,
        "header": None,
        "column_count": None,
        "features": None,
        "dtypes": None,
        "row_count": None,
        "row_count_approximate": None,
    }


def _tabular_schema(files: List[Path], max_file_bytes: int, reads: Dict[str, int]) -> dict:
    """Schema metadata for a site: names/dtypes from the first file, rows from all."""
    tabular_files = [f for f in files if f.suffix.lower() in TABULAR_EXTENSIONS]
    if not tabular_files:
        # e.g. an image-only site under a tabular-majority dataset: no schema
        # keys at all rather than a parquet-shaped block of nulls
        return {}
    text_files = [f for f in tabular_files if f.suffix.lower() in TEXT_TABULAR_EXTENSIONS]
    if not text_files:
        return _parquet_schema(tabular_files, reads)
    target = text_files[0]
    schema = _null_schema(target.suffix.lower().lstrip("."))
    raw, capped = _read_bounded(target, max_file_bytes, reads)
    if raw is None:
        return schema
    text = raw.decode("utf-8", errors="replace")
    delimiter = "\t" if target.suffix.lower() == ".tsv" else ","
    rows = []
    for row in csv.reader(io.StringIO(text), delimiter=delimiter):
        if row:
            rows.append(row[:MAX_SCHEMA_COLUMNS])
        if len(rows) >= MAX_SAMPLE_ROWS:
            break
    if len(rows) < 2:
        return schema

    column_count = len(rows[0])
    body = [r for r in rows[1:] if len(r) == column_count]
    numeric = [_column_is_numeric(body, j) for j in range(column_count)]
    header = HEADER_AMBIGUOUS
    for j in range(column_count):
        if numeric[j] and not _numeric_token(rows[0][j]):
            header = HEADER_PRESENT
            break

    features = None
    names_truncated = False
    if header == HEADER_PRESENT:
        features, names_truncated = _sanitize_names(rows[0])

    # rows in the schema file, within the byte cap
    data_rows = len(rows) - (1 if header == HEADER_PRESENT else 0)
    if len(rows) >= MAX_SAMPLE_ROWS or capped:
        data_rows = max(data_rows, text.count("\n") - (1 if header == HEADER_PRESENT else 0))
    approximate = capped
    shards_consistent = True
    # sharded site: aggregate line counts across the remaining files; per-shard
    # header presence is unknowable, so any multi-file total is approximate.
    # A shard whose first row has a different field count disagrees with the
    # site schema and must not stay silent.
    for extra in text_files[1:]:
        extra_raw, _extra_capped = _read_bounded(extra, max_file_bytes, reads)
        if extra_raw is not None:
            extra_text = extra_raw.decode("utf-8", errors="replace")
            data_rows += extra_text.count("\n")
            first = next(csv.reader(io.StringIO(extra_text), delimiter=delimiter), None)
            if first and len(first[:MAX_SCHEMA_COLUMNS]) != column_count:
                shards_consistent = False
        approximate = True
    # a site mixing text and parquet formats is counted from the text files
    # only; never present that as an exact total
    if len(tabular_files) > len(text_files):
        approximate = True
        shards_consistent = False

    schema.update(
        {
            "header": header,
            "column_count": column_count,
            "features": features,
            "dtypes": ["numeric" if numeric[j] else "text" for j in range(column_count)],
            "row_count": data_rows,
            "row_count_approximate": approximate,
        }
    )
    if names_truncated:
        schema["feature_names_truncated"] = True
    if not shards_consistent:
        schema["shard_schema_consistent"] = False
    return schema


def _sanitize_names(cells: List[str]) -> Tuple[List[str], bool]:
    """Bound emitted names: strip control characters, cap the length."""
    names = []
    truncated = False
    for cell in cells:
        cleaned = "".join(ch for ch in cell.strip() if ch.isprintable())
        if len(cleaned) > MAX_FEATURE_NAME_CHARS:
            cleaned = cleaned[:MAX_FEATURE_NAME_CHARS]
            truncated = True
        names.append(cleaned)
    return names, truncated


def _read_bounded(path: Path, max_file_bytes: int, reads: Dict[str, int]):
    try:
        raw = path.open("rb").read(max_file_bytes)
    except OSError:
        return None, False
    reads["files_read"] += 1
    reads["bytes_read"] += len(raw)
    return raw, len(raw) >= max_file_bytes


def _parquet_schema(files: List[Path], reads: Dict[str, int]) -> dict:
    """Parquet metadata via optional pyarrow; explicit fallback marker without it.

    Row counts sum the footer metadata of every parquet shard in the site, so
    sharded parquet sites are exact; a shard whose footer cannot be read makes
    the total approximate.
    """
    schema = _null_schema("parquet")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        # No reader available: the caller must obtain declared names or fail
        # closed, exactly like an ambiguous header.
        schema["schema_available"] = False
        return schema

    parquet_files = [f for f in files if f.suffix.lower() == ".parquet"]
    features = None
    dtypes = None
    reference_signature = None
    total_rows = 0
    approximate = False
    shards_consistent = True
    for target in parquet_files:
        try:
            parquet_file = pq.ParquetFile(target)
        except Exception:
            approximate = True
            continue
        reads["files_read"] += 1
        reads["bytes_read"] += target.stat().st_size  # footer read; size is the upper bound
        total_rows += parquet_file.metadata.num_rows
        arrow_schema = parquet_file.schema_arrow
        shard_names, _ = _sanitize_names(list(arrow_schema.names)[:MAX_SCHEMA_COLUMNS])
        shard_dtypes = [
            "numeric" if (pa.types.is_integer(field.type) or pa.types.is_floating(field.type)) else "text"
            for field in arrow_schema
        ][:MAX_SCHEMA_COLUMNS]
        signature = (tuple(shard_names), tuple(shard_dtypes))
        if features is None:
            features, dtypes, reference_signature = shard_names, shard_dtypes, signature
        elif signature != reference_signature:
            # shards within one site disagree: a summed row count across
            # different schemas is not exact, and the disagreement must
            # surface in schema_agreement rather than stay silent
            shards_consistent = False
            approximate = True

    if features is None:
        schema["schema_available"] = False
        schema["row_count_approximate"] = approximate or None
        return schema
    schema.update(
        {
            "schema_available": True,
            "header": HEADER_PRESENT,  # means "names available"; parquet always carries names
            "column_count": len(features),
            "features": features,
            "dtypes": dtypes,
            "row_count": total_rows,
            "row_count_approximate": approximate,
        }
    )
    if not shards_consistent:
        schema["shard_schema_consistent"] = False
    return schema


def _column_is_numeric(body: List[List[str]], index: int) -> bool:
    values = [row[index].strip() for row in body]
    non_empty = [v for v in values if v]
    return bool(non_empty) and all(_numeric_token(v) for v in non_empty)


def _numeric_token(value: str) -> bool:
    if value.strip().lower() in _NON_NUMERIC_FLOAT_TOKENS:
        return False
    try:
        float(value)
        return True
    except ValueError:
        return False


def _schema_agreement(sites: List[dict]) -> dict:
    """Compare names, column counts, AND dtype classes across sites.

    Same feature names with drifting dtypes (income numeric at one site, text
    at another) is not analysis-ready for federated statistics, so dtype
    drift is a mismatch, not a footnote.
    """
    reference = None
    mismatches = []
    for site in sites:
        if site.get("shard_schema_consistent") is False:
            mismatches.append({"site": site["name"], "reference_site": site["name"], "issue": "shards_differ"})
        names = tuple(site["features"]) if site.get("features") else None
        signature = (names or site.get("column_count"), tuple(site["dtypes"]) if site.get("dtypes") else None)
        if signature[0] is None:
            continue
        if reference is None:
            reference = (site["name"], signature)
            continue
        if signature == reference[1]:
            continue
        if signature[0] != reference[1][0]:
            issue = "feature_names_differ" if isinstance(signature[0], tuple) else "column_count_differs"
        else:
            issue = "dtypes_differ"
        mismatches.append({"site": site["name"], "reference_site": reference[0], "issue": issue})
    return {"status": "mismatch" if mismatches else "consistent", "mismatches": mismatches}


def _sample_pixel_depth(files: List[Path], reads: Dict[str, int]) -> Optional[str]:
    """Bit-depth class of one sample image, when an optional loader exists."""
    try:
        from PIL import Image
    except ImportError:
        return None
    for path in files:
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        try:
            with Image.open(path) as img:
                mode = img.mode
            reads["files_read"] += 1
            # PIL reads lazily; the file size is the upper bound of the read
            reads["bytes_read"] += path.stat().st_size
            return {"L": "uint8", "RGB": "uint8", "RGBA": "uint8", "I;16": "uint16", "I": "int32", "F": "float"}.get(
                mode, mode
            )
        except Exception:
            continue
    return None
