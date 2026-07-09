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
metadata-only evidence block: site layout, feature names and inferred dtype
classes, row/file counts, and cross-site schema agreement. It never emits
cell values or pixel data, follows no symlinks, and bounds every read, so
it preserves the inspector's static/redaction posture. Skills consume this
block instead of hand-rolling their own data inspection.
"""

import csv
import io
from pathlib import Path
from typing import Dict, List, Optional

TABULAR_EXTENSIONS = {".csv", ".tsv", ".parquet"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".nii"}
MAX_SAMPLE_ROWS = 200
MAX_SCHEMA_COLUMNS = 512
MAX_WALK_ENTRIES = 50_000

HEADER_PRESENT = "present"
# Per the fed-stats contract: a first row is a header only when at least one
# column shows text over an otherwise-numeric column. Anything else (all
# numeric, or text over text columns) cannot be distinguished from data, so
# the caller must obtain declared names or an explicit header statement.
HEADER_AMBIGUOUS = "ambiguous"


def inspect_dataset(root: Path, max_files: int, max_file_bytes: int) -> Optional[dict]:
    """Classify a directory as a dataset; None when it holds no data files."""
    groups, census, truncated = _collect_data_files(root, max_files)
    tabular_count = sum(census.get(ext, 0) for ext in TABULAR_EXTENSIONS)
    image_count = sum(n for ext, n in census.items() if ext in IMAGE_EXTENSIONS or ext == ".nii.gz")
    if tabular_count == 0 and image_count == 0:
        return None
    if tabular_count == image_count:
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
            schema = _tabular_schema(files, max_file_bytes, reads)
            if schema is not None:
                site.update(schema)
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

    The ``max_files`` limit counts *data* files, so non-data clutter cannot
    exhaust the budget before any data file is seen; ``MAX_WALK_ENTRIES``
    separately bounds total traversal work.
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
            if child.is_dir():
                stack.append(child)
                continue
            if not child.is_file():
                continue
            entries_seen += 1
            if entries_seen > MAX_WALK_ENTRIES:
                truncated = True
                stack.clear()
                break
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


def _tabular_schema(files: List[Path], max_file_bytes: int, reads: Dict[str, int]) -> Optional[dict]:
    """Schema metadata for a site: names/dtypes from the first file, rows from all."""
    text_files = [f for f in files if f.suffix.lower() in {".csv", ".tsv"}]
    if not text_files:
        return _parquet_schema(files, reads)
    target = text_files[0]
    raw, capped = _read_bounded(target, max_file_bytes, reads)
    if raw is None:
        return None
    text = raw.decode("utf-8", errors="replace")
    delimiter = "\t" if target.suffix.lower() == ".tsv" else ","
    rows = []
    for row in csv.reader(io.StringIO(text), delimiter=delimiter):
        if row:
            rows.append(row[:MAX_SCHEMA_COLUMNS])
        if len(rows) >= MAX_SAMPLE_ROWS:
            break
    if len(rows) < 2:
        return {"format": target.suffix.lower().lstrip("."), "header": None, "features": None, "dtypes": None}

    column_count = len(rows[0])
    body = [r for r in rows[1:] if len(r) == column_count]
    numeric = [_column_is_numeric(body, j) for j in range(column_count)]
    header = HEADER_AMBIGUOUS
    for j in range(column_count):
        if numeric[j] and not _parses_as_float(rows[0][j]):
            header = HEADER_PRESENT
            break

    dtypes = ["numeric" if numeric[j] else "text" for j in range(column_count)]
    features = [cell.strip() for cell in rows[0]] if header == HEADER_PRESENT else None
    # rows in the first file, within the byte cap
    data_rows = len(rows) - (1 if header == HEADER_PRESENT else 0)
    if len(rows) >= MAX_SAMPLE_ROWS or capped:
        data_rows = max(data_rows, text.count("\n") - (1 if header == HEADER_PRESENT else 0))
    approximate = capped
    # a sharded site: aggregate line counts across the remaining files; the
    # per-shard header presence is unknowable, so the total is approximate
    for extra in text_files[1:]:
        extra_raw, extra_capped = _read_bounded(extra, max_file_bytes, reads)
        if extra_raw is None:
            approximate = True
            continue
        data_rows += extra_raw.decode("utf-8", errors="replace").count("\n")
        approximate = True
        if extra_capped:
            approximate = True
    return {
        "format": target.suffix.lower().lstrip("."),
        "header": header,
        "column_count": column_count,
        "features": features,
        "dtypes": dtypes,
        "row_count": data_rows,
        "row_count_approximate": approximate,
    }


def _read_bounded(path: Path, max_file_bytes: int, reads: Dict[str, int]):
    try:
        raw = path.open("rb").read(max_file_bytes)
    except OSError:
        return None, False
    reads["files_read"] += 1
    reads["bytes_read"] += len(raw)
    return raw, len(raw) >= max_file_bytes


def _parquet_schema(files: List[Path], reads: Dict[str, int]) -> dict:
    """Parquet metadata via optional pyarrow; explicit fallback marker without it."""
    base = {"format": "parquet", "header": None, "features": None, "dtypes": None}
    try:
        import pyarrow.parquet as pq
    except ImportError:
        # No reader available: the caller must obtain declared names or fail
        # closed, exactly like an ambiguous header.
        base["schema_available"] = False
        return base
    target = next((f for f in files if f.suffix.lower() == ".parquet"), None)
    if target is None:
        base["schema_available"] = False
        return base
    try:
        parquet_file = pq.ParquetFile(target)
        schema = parquet_file.schema_arrow
        row_count = parquet_file.metadata.num_rows
    except Exception:
        base["schema_available"] = False
        return base
    reads["files_read"] += 1
    reads["bytes_read"] += target.stat().st_size  # footer read; size is the upper bound
    dtypes = []
    import pyarrow as pa

    for field in schema:
        is_numeric = pa.types.is_integer(field.type) or pa.types.is_floating(field.type)
        dtypes.append("numeric" if is_numeric else "text")
    base.update(
        {
            "schema_available": True,
            "header": HEADER_PRESENT,
            "column_count": len(schema.names),
            "features": list(schema.names)[:MAX_SCHEMA_COLUMNS],
            "dtypes": dtypes[:MAX_SCHEMA_COLUMNS],
            "row_count": row_count,
            "row_count_approximate": False,
        }
    )
    return base


def _column_is_numeric(body: List[List[str]], index: int) -> bool:
    values = [row[index].strip() for row in body]
    non_empty = [v for v in values if v]
    return bool(non_empty) and all(_parses_as_float(v) for v in non_empty)


def _parses_as_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _schema_agreement(sites: List[dict]) -> dict:
    """Compare feature names (or column counts) across sites; names, no values."""
    reference = None
    mismatches = []
    for site in sites:
        signature = (site.get("features") and tuple(site["features"])) or site.get("column_count")
        if signature is None:
            continue
        if reference is None:
            reference = (site["name"], signature)
            continue
        if signature != reference[1]:
            mismatches.append(
                {
                    "site": site["name"],
                    "reference_site": reference[0],
                    "issue": "feature_names_differ" if isinstance(signature, tuple) else "column_count_differs",
                }
            )
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
