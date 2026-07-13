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
metadata-only evidence block. It never emits values from rows classified
as data and never emits pixel data, and follows no symlinks. The one
inference in the block is the header row: names are emitted only under
the strong rule (every numeric-bodied column non-numeric, names unique
and non-empty), and a masked data row that satisfies all of that is
emitted as names — by construction no inference can tell it from a
header, so datasets with masked/sentinel first rows must declare names. Read bounds: text and image reads are capped at
``max_file_bytes`` each (worst case ``max_files x max_file_bytes``);
parquet metadata reads parse only the file footer via pyarrow — they are
not capped by ``max_file_bytes`` and are accounted in ``scan`` at file
size as the upper bound.

Dataset block contract (all keys always present per modality):

- ``modality``: ``tabular`` | ``image`` | ``mixed``, decided per site, not
  by file-count share. Targeted shapes: a tabular dataset tolerates stray
  images (<= 2 per site, e.g. exported plots); an image dataset tolerates
  companion tabular metadata (<= 4 files per site, e.g. train/val/test
  label files, flagged ``tabular_companions``, never a statistics
  target); tiny sites inside both tolerance windows resolve to the side
  with more files overall; anything else — an image-only site among
  tabular sites, materially both modalities, or a dead file-count tie —
  is ``mixed``, reported but not routed (``target_type`` stays
  ``unknown_target``).
- ``layout``: ``per_site_directories`` | ``flat`` |
  ``root_and_site_directories`` (root-level data files coexist with site
  directories — ambiguous site mapping, an ask/fail-closed input for the
  consumer; the root group appears as site ``.``).
- ``file_census``: extension -> count (bounded; see ``counts_approximate``).
- ``counts_approximate``: true when a walk limit was hit.
- ``scan``: ``files_read`` / ``bytes_read`` performed by this classification.
- ``sites``: per top-level directory (or ``.`` when flat):
  - always: ``name``, ``data_files``, ``tabular_files``, ``image_files``;
  - tabular/parquet: ``format``, ``header``, ``column_count``, ``features``,
    ``dtypes``, ``row_count``, ``row_count_approximate``, and for parquet
    ``schema_available``. ``header: present`` means "names available"
    (parquet always has names when readable); ``header: ambiguous`` means
    names must be user-supplied. ``row_count`` aggregates every tabular
    file in the site; sharded CSV sites are always approximate (per-shard
    headers are unknowable; a shard whose header-like first row disagrees
    with the site schema flags ``shard_schema_consistent: false``), parquet
    shards are exact metadata sums when the shards agree on schema.
    Feature names are sanitized: control characters stripped and length
    capped (``feature_names_truncated`` marks a cap hit); schemas wider
    than the column cap set ``columns_truncated`` — drift past the cap is
    invisible. ``header: present`` requires every numeric-bodied column
    to carry a non-numeric first value AND the names to be unique and
    non-empty; parquet names failing that are flagged
    ``feature_names_invalid``. Dtype classes
    are inferred from a bounded row sample; a full-file reader (pandas at
    job runtime) can disagree when anomalies appear past the sample.
  - image: ``image_formats`` (the site's image extensions, so the consumer
    picks the matching loader) and ``pixel_depth`` from one sample when an
    optional loader exists (PIL formats only; DICOM/NIfTI depth stays
    null and the loader must be chosen from ``image_formats``).
- ``schema_agreement`` (tabular): compares feature names, column counts,
  AND dtype classes across sites; any drift is a ``mismatch`` with per-site
  issues (``feature_names_differ`` | ``column_count_differs`` |
  ``header_presence_differs`` (one site has a header, another is ambiguous
  — alignment unverifiable even at equal widths) | ``dtypes_differ`` |
  ``shards_differ``, the last when shards inside one site disagree —
  flagged per site as ``shard_schema_consistent: false``).

Walk bounds: ``max_files`` counts data files so non-data clutter cannot
hide a dataset, and ``MAX_WALK_ENTRIES`` bounds total traversal (files and
directories). This deliberately differs from the code walk in
``inspector.py``, which bounds per-file scan work; the two walks bound
different costs and are kept separate on purpose.
"""

import csv
import io
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TABULAR_EXTENSIONS = {".csv", ".tsv", ".parquet"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".nii"}
TEXT_TABULAR_EXTENSIONS = {".csv", ".tsv"}
MAX_SAMPLE_ROWS = 200
MAX_SCHEMA_COLUMNS = 512
MAX_FEATURE_NAME_CHARS = 120
MAX_WALK_ENTRIES = 50_000
# Per-site structural modality rules (file-count percentages misjudge mixed
# data: one CSV can be a complete dataset while image datasets inherently
# hold hundreds of files):
# - a tabular site tolerates at most this many stray images (exported plots);
STRAY_MAX_IMAGES_PER_SITE = 2
# - an image site tolerates at most this many tabular files as companion
#   metadata (train/val/test label files beside the scans are common),
#   reported but never a stats target.
COMPANION_MAX_TABULAR_PER_SITE = 4

HEADER_PRESENT = "present"
# Per the fed-stats contract: a first row is a header only when EVERY
# numeric-bodied column carries a non-numeric first value (weaker evidence,
# like one text token over one numeric column, can be a masked data row and
# emitting it would leak cell values as feature names). Anything else cannot
# be distinguished from data, so the caller must obtain declared names or an
# explicit header statement. Residual risk: a first data row that is
# non-numeric in every numeric column is indistinguishable from a header by
# construction - datasets with masked/sentinel first rows should declare
# names.
HEADER_AMBIGUOUS = "ambiguous"

# float() accepts nan/inf tokens; a column of literal "nan" strings is not
# numeric evidence for statistics.
_NON_NUMERIC_FLOAT_TOKENS = {"nan", "inf", "-inf", "+inf", "infinity", "-infinity", "+infinity"}


def inspect_dataset(root: Path, max_files: int, max_file_bytes: int) -> Optional[dict]:
    """Classify a directory as a dataset; None when it holds no data files."""
    groups, census, truncated = _collect_data_files(root, max_files)
    if not groups:
        return None
    modality = _dataset_modality(groups)

    # Dataset classification reads data bytes (bounded, metadata-only);
    # account for every read so the inspection stays auditable.
    reads = {"files_read": 0, "bytes_read": 0}
    sites = []
    for name in sorted(groups):
        files = groups[name]
        tabular_files = [f for f in files if _data_extension(f) in TABULAR_EXTENSIONS]
        image_count = len(files) - len(tabular_files)
        site: Dict = {
            "name": name,
            "data_files": len(files),
            "tabular_files": len(tabular_files),
            "image_files": image_count,
        }
        if modality == "tabular":
            site.update(_tabular_schema(files, max_file_bytes, reads))
        elif modality == "image":
            site["image_formats"] = sorted(
                {_data_extension(f) for f in files if _data_extension(f) not in TABULAR_EXTENSIONS} - {None}
            )
            site["pixel_depth"] = _sample_pixel_depth(files, reads)
            if tabular_files:
                # companion metadata (e.g. labels.csv): reported, never a
                # statistics target
                site["tabular_companions"] = len(tabular_files)
        sites.append(site)

    site_names = {s["name"] for s in sites}
    if site_names == {"."}:
        layout = "flat"
    elif "." in site_names:
        # root-level data files coexist with site directories: the site
        # mapping is ambiguous and the consumer must resolve it explicitly
        layout = "root_and_site_directories"
    else:
        layout = "per_site_directories"
    dataset: Dict = {
        "modality": modality,
        "layout": layout,
        "file_census": {ext: census[ext] for ext in sorted(census)},
        "counts_approximate": truncated,
        "scan": reads,
        "sites": sites,
    }
    if modality == "tabular":
        dataset["schema_agreement"] = _schema_agreement(sites)
    return dataset


def _dataset_modality(groups: Dict[str, List[Path]]) -> str:
    """Per-site structural classification; see the module contract.

    tabular-consistent: every site has tabular files and at most
    ``STRAY_MAX_IMAGES_PER_SITE`` images. image-consistent: every site has
    image files and at most ``COMPANION_MAX_TABULAR_PER_SITE`` tabular
    files. Exactly one consistent view wins. When tiny sites satisfy both
    tolerance windows, the side with more files overall is the substantive
    dataset and a dead tie is ``mixed``; neither view consistent (an
    image-only site among tabular sites, or materially both) is ``mixed``
    and stays unrouted.
    """
    tabular_ok = True
    image_ok = True
    total_tabular = 0
    total_images = 0
    for files in groups.values():
        tabular = sum(1 for f in files if _data_extension(f) in TABULAR_EXTENSIONS)
        images = len(files) - tabular
        total_tabular += tabular
        total_images += images
        if tabular == 0 or images > STRAY_MAX_IMAGES_PER_SITE:
            tabular_ok = False
        if images == 0 or tabular > COMPANION_MAX_TABULAR_PER_SITE:
            image_ok = False
    if tabular_ok and image_ok:
        # tiny sites sit inside both tolerance windows (a few files of
        # each); the substantive side is whichever holds more files, and a
        # dead tie is genuinely ambiguous
        if total_tabular == total_images:
            return "mixed"
        return "tabular" if total_tabular > total_images else "image"
    if not tabular_ok and not image_ok:
        return "mixed"
    return "tabular" if tabular_ok else "image"


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
        # consume a bounded number of entries BEFORE sorting, so a huge
        # directory cannot defeat the traversal cap with an unbounded
        # listing; within the budget the sorted order stays deterministic
        budget = MAX_WALK_ENTRIES - entries_seen + 1
        try:
            raw_children = list(islice(directory.iterdir(), max(budget, 0)))
        except OSError:
            continue
        children = sorted(raw_children, key=lambda p: p.name, reverse=True)
        for child in children:
            # symlinks consume walk budget too (they are traversal work),
            # they are just never followed
            entries_seen += 1
            if entries_seen > MAX_WALK_ENTRIES:
                truncated = True
                stack.clear()
                break
            if child.is_symlink():
                continue
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
    columns_truncated = False
    for row in csv.reader(io.StringIO(text), delimiter=delimiter):
        if row:
            if len(row) > MAX_SCHEMA_COLUMNS:
                columns_truncated = True
            rows.append(row[:MAX_SCHEMA_COLUMNS])
        if len(rows) >= MAX_SAMPLE_ROWS:
            break
    if len(rows) < 2:
        return schema

    column_count = len(rows[0])
    body = [r for r in rows[1:] if len(r) == column_count]
    numeric = [_column_is_numeric(body, j) for j in range(column_count)]
    # Header evidence must be strong: EVERY numeric-bodied column carries a
    # non-numeric first value. A single text token over one numeric column
    # (e.g. a masked first data row like "SUPPRESSED,100") is NOT a header -
    # emitting it would leak cell values as feature names.
    numeric_columns = [j for j in range(column_count) if numeric[j]]
    features = None
    names_truncated = False
    header = HEADER_AMBIGUOUS
    if numeric_columns and all(not _numeric_token(rows[0][j]) for j in numeric_columns):
        candidate, names_truncated = _sanitize_names(rows[0])
        # a valid header must also name every column uniquely: duplicates or
        # empties (e.g. a masked data row "SUPPRESSED,SUPPRESSED") are not
        # headers and must not be emitted as names
        if all(candidate) and len(set(candidate)) == len(candidate):
            header = HEADER_PRESENT
            features = candidate

    # rows in the schema file, within the byte cap; the line-count fallback
    # adds the final unterminated line so uncapped counts stay exact
    data_rows = len(rows) - (1 if header == HEADER_PRESENT else 0)
    if len(rows) >= MAX_SAMPLE_ROWS or capped:
        line_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        data_rows = max(data_rows, line_count - (1 if header == HEADER_PRESENT else 0))
    approximate = capped
    shards_consistent = True
    # sharded site: aggregate line counts across the remaining files; per-shard
    # header presence is unknowable, so any multi-file total is approximate.
    # A shard disagrees when its first row has a different field count, or
    # when its first row is header-like (text where the site dtypes are
    # numeric) but carries different names than the site schema.
    for extra in text_files[1:]:
        extra_raw, _extra_capped = _read_bounded(extra, max_file_bytes, reads)
        if extra_raw is not None:
            extra_text = extra_raw.decode("utf-8", errors="replace")
            data_rows += extra_text.count("\n")
            first = next(csv.reader(io.StringIO(extra_text), delimiter=delimiter), None)
            if first:
                first = first[:MAX_SCHEMA_COLUMNS]
                if len(first) != column_count:
                    shards_consistent = False
                elif header == HEADER_PRESENT and _row_is_header_like(first, dtypes_flags=numeric):
                    shard_names, _ = _sanitize_names(first)
                    if shard_names != features:
                        shards_consistent = False
                    else:
                        # a detected repeated header is not a data row
                        data_rows -= 1
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
    if columns_truncated:
        schema["columns_truncated"] = True
    if not shards_consistent:
        schema["shard_schema_consistent"] = False
    return schema


def _row_is_header_like(row: List[str], dtypes_flags: List[bool]) -> bool:
    """True when a row carries text in columns the site knows are numeric."""
    return any(flag and not _numeric_token(cell) for flag, cell in zip(dtypes_flags, row))


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
        with path.open("rb") as f:
            raw = f.read(max_file_bytes)
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
        if len(arrow_schema.names) > MAX_SCHEMA_COLUMNS:
            schema["columns_truncated"] = True
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
    if not all(features) or len(set(features)) != len(features):
        # real metadata, but pathological: duplicate or empty column names
        # will be mangled by downstream readers
        schema["feature_names_invalid"] = True
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
    drift is a mismatch, not a footnote. A header-bearing site next to a
    header-ambiguous one is also a mismatch even at equal widths — column
    alignment cannot be verified without names on both sides — labeled
    ``header_presence_differs`` so it is not mistaken for a width drift.
    """
    reference = None
    mismatches = []
    for site in sites:
        if site.get("shard_schema_consistent") is False:
            mismatches.append({"site": site["name"], "reference_site": site["name"], "issue": "shards_differ"})
        names = tuple(site["features"]) if site.get("features") else None
        count = len(names) if names is not None else site.get("column_count")
        dtypes = tuple(site["dtypes"]) if site.get("dtypes") else None
        if count is None:
            continue
        signature = (names, count, dtypes)
        if reference is None:
            reference = (site["name"], signature)
            continue
        if signature == reference[1]:
            continue
        ref_names, ref_count, _ = reference[1]
        if count != ref_count:
            issue = "column_count_differs"
        elif (names is None) != (ref_names is None):
            issue = "header_presence_differs"
        elif names != ref_names:
            issue = "feature_names_differ"
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
