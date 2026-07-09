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

    sites = []
    for name in sorted(groups):
        files = groups[name]
        site: Dict = {"name": name, "data_files": len(files)}
        if modality == "tabular":
            schema = _tabular_schema(files, max_file_bytes)
            if schema is not None:
                site.update(schema)
        elif modality == "image":
            site["pixel_depth"] = _sample_pixel_depth(files)
        sites.append(site)

    dataset: Dict = {
        "modality": modality,
        "layout": "per_site_directories" if any(s["name"] != "." for s in sites) else "flat",
        "file_census": {ext: census[ext] for ext in sorted(census)},
        "counts_approximate": truncated,
        "sites": sites,
    }
    if modality == "tabular":
        dataset["schema_agreement"] = _schema_agreement(sites)
    return dataset


def _collect_data_files(root: Path, max_files: int):
    """Bounded, sorted, symlink-free walk grouping data files by top-level dir."""
    groups: Dict[str, List[Path]] = {}
    census: Dict[str, int] = {}
    visited = 0
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
            visited += 1
            if visited > max_files:
                truncated = True
                return groups, census, truncated
            ext = _data_extension(child)
            if ext is None:
                continue
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


def _tabular_schema(files: List[Path], max_file_bytes: int) -> Optional[dict]:
    """Schema metadata for a site's first tabular file: names/dtypes, no values."""
    target = next((f for f in files if f.suffix.lower() in {".csv", ".tsv"}), None)
    if target is None:
        # parquet-only site: schema needs an optional reader; report format only
        return {"format": "parquet", "header": None, "features": None, "dtypes": None}
    try:
        raw = target.open("rb").read(max_file_bytes)
    except OSError:
        return None
    capped = len(raw) >= max_file_bytes
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
    # row count within the byte cap; approximate when the cap was hit
    data_rows = len(rows) - (1 if header == HEADER_PRESENT else 0)
    if len(rows) >= MAX_SAMPLE_ROWS or capped:
        data_rows = max(data_rows, text.count("\n") - (1 if header == HEADER_PRESENT else 0))
    return {
        "format": target.suffix.lower().lstrip("."),
        "header": header,
        "column_count": column_count,
        "features": features,
        "dtypes": dtypes,
        "row_count": data_rows,
        "row_count_approximate": capped,
    }


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


def _sample_pixel_depth(files: List[Path]) -> Optional[str]:
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
            return {"L": "uint8", "RGB": "uint8", "RGBA": "uint8", "I;16": "uint16", "I": "int32", "F": "float"}.get(
                mode, mode
            )
        except Exception:
            continue
    return None
