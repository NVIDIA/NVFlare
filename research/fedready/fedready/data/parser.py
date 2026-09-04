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

"""Safe local dataset profiling for FedReady client agents.

The parser is intended to run on a client. It summarizes a local dataset into
aggregate metadata that a client may share with the server during FL task
negotiation. Raw file paths, filenames, patient identifiers, and per-sample
records are never included in the returned profile.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import tarfile
import zipfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator

from fedready.utils.io import atomic_write_json

try:  # Pillow is optional, but available in the vlm_env used for this project.
    from PIL import Image
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    Image = None  # type: ignore[assignment]


SCHEMA_VERSION = "fedready.local_dataset_profile.v1"
SITE_CLIENTS_SCHEMA_VERSION = "fedready.site_clients.v1"
CLIENT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")

IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}
TABULAR_EXTENSIONS = {".csv", ".tsv", ".xls", ".xlsx"}
ANNOTATION_EXTENSIONS = TABULAR_EXTENSIONS | {".json", ".xml", ".txt"}
CSV_EXTENSIONS = {".csv", ".tsv"}

SENSITIVE_ANNOTATION_COLUMN_TOKENS = {
    "accession",
    "annotator",
    "author",
    "birth",
    "center",
    "centre",
    "clinician",
    "contact",
    "creator",
    "date",
    "dob",
    "doctor",
    "email",
    "firstname",
    "first_name",
    "fullname",
    "full_name",
    "grader",
    "hospital",
    "id",
    "identifier",
    "institution",
    "lastname",
    "last_name",
    "mrn",
    "name",
    "operator",
    "organization",
    "patient",
    "physician",
    "reader",
    "record",
    "reviewer",
    "scan_date",
    "site",
    "subject",
    "time",
    "timestamp",
}

LABEL_COLUMN_EXACT_NAMES = {
    "category",
    "category_name",
    "class",
    "class_label",
    "class_name",
    "classification",
    "classification_label",
    "diagnosis",
    "diagnosis_code",
    "diagnosis_label",
    "diagnosis_name",
    "diagnostic_label",
    "disease",
    "disease_grade",
    "disease_label",
    "disease_name",
    "grade",
    "label",
    "label_name",
    "level",
    "severity",
    "stage",
    "target",
    "target_label",
    "target_name",
}
LABEL_COLUMN_CORE_TOKENS = {
    "category",
    "class",
    "classification",
    "diagnosis",
    "diagnostic",
    "disease",
    "grade",
    "label",
    "level",
    "severity",
    "stage",
    "status",
    "target",
    "type",
}
LABEL_COLUMN_SAFE_MODIFIER_TOKENS = {
    "binary",
    "code",
    "dr",
    "final",
    "gt",
    "integer",
    "map",
    "mapping",
    "multi",
    "multiclass",
    "primary",
    "secondary",
    "truth",
    "value",
}
LABEL_VALUE_DOMAIN_TOKENS = {
    "amd",
    "background",
    "cataract",
    "central",
    "chorioretinopathy",
    "control",
    "cup",
    "diabetic",
    "disc",
    "disease",
    "diseased",
    "dme",
    "dr",
    "edema",
    "exudate",
    "glaucoma",
    "healthy",
    "hemorrhage",
    "hypertension",
    "hypertensive",
    "lesion",
    "macular",
    "microaneurysm",
    "mild",
    "moderate",
    "myopia",
    "negative",
    "no",
    "normal",
    "nonproliferative",
    "npdr",
    "optic",
    "pdr",
    "positive",
    "proliferative",
    "retina",
    "retinal",
    "retinopathy",
    "serous",
    "severe",
    "target",
    "toxoplasmosis",
    "uveitis",
    "vessel",
}

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "tr": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
    "testing": "test",
    "ts": "test",
}

MASK_HINTS = {
    "annotation",
    "annotations",
    "gt",
    "ground",
    "groundtruth",
    "ground_truth",
    "label",
    "labels",
    "manual",
    "mask",
    "masks",
    "seg",
    "segmentation",
}

SEGMENTATION_ANNOTATION_HINTS = {
    "anotation",
    "annotation",
    "annotations",
    "boundary",
    "boundaries",
    "contour",
    "contours",
    "expert",
    "experts",
    "ground",
    "groundtruth",
    "ground_truth",
    "seg",
    "segmentation",
    "segmentations",
}


GENERIC_LABEL_TOKENS = {
    "archive",
    "archives",
    "benchmark",
    "benchmarks",
    "case",
    "cases",
    "channel",
    "channels",
    "data",
    "dataset",
    "datasets",
    "and",
    "for",
    "from",
    "file",
    "files",
    "fit",
    "fits",
    "image",
    "images",
    "img",
    "jpeg",
    "jpg",
    "label",
    "labels",
    "mask",
    "masks",
    "png",
    "raw",
    "reference",
    "references",
    "segmentation",
    "segmentations",
    "the",
    "to",
    "with",
    "test",
    "train",
    "training",
    "val",
    "valid",
    "validation",
}


@dataclass(frozen=True)
class DataParserConfig:
    """Controls bounded local dataset profiling."""

    min_count: int = 5
    max_scan_files: int = 200_000
    max_image_samples: int = 200
    histogram_bins: int = 16
    histogram_max_side: int = 256
    max_annotation_files: int = 20
    max_annotation_bytes: int = 2_000_000
    max_annotation_rows: int = 5_000
    include_archives: bool = True

    def __post_init__(self) -> None:
        if self.min_count < 1:
            raise ValueError("min_count must be at least 1")
        if self.max_scan_files < 1:
            raise ValueError("max_scan_files must be at least 1")
        if self.max_image_samples < 1:
            raise ValueError("max_image_samples must be at least 1")
        if not 1 <= self.histogram_bins <= 256:
            raise ValueError("histogram_bins must be between 1 and 256")
        if self.histogram_max_side < 16:
            raise ValueError("histogram_max_side must be at least 16")


@dataclass(frozen=True)
class FileRecord:
    storage: str
    container_path: Path
    logical_path: str
    extension: str
    size_bytes: int | None = None
    member_name: str | None = None


@dataclass(frozen=True)
class ScanResult:
    records: list[FileRecord]
    archive_count: int
    local_file_count: int
    archive_member_count: int
    scan_truncated: bool
    warnings: list[str]
    client_identity_scope_applied: bool = False


def parse_dataset(
    data_path: str | Path,
    *,
    client_id: str | None = None,
    config: DataParserConfig | None = None,
) -> dict[str, Any]:
    """Profile one local dataset using only aggregate safe-to-share metadata."""

    cfg = config or DataParserConfig()
    root = Path(data_path)
    data_card = _load_data_card(root)
    scan = _scope_scan_to_client_subdataset(_scan_dataset(root, cfg), client_id=client_id)

    image_records = [record for record in scan.records if _is_image_record(record)]
    mask_records = [record for record in image_records if _is_mask_record(record)]
    mask_record_set = set(mask_records)
    primary_image_records = [record for record in image_records if record not in mask_record_set]
    if not primary_image_records:
        primary_image_records = image_records

    split_counts = _count_splits(primary_image_records)
    class_labels = _infer_class_labels(primary_image_records, cfg)
    annotation_profile = _profile_annotations(scan.records, cfg)
    label_profile = _profile_labels(
        data_card, mask_records, class_labels, annotation_profile, primary_image_records, cfg
    )
    image_profile = _profile_images(primary_image_records, cfg)

    tabular_count = sum(1 for record in scan.records if record.extension in TABULAR_EXTENSIONS)
    annotation_count = sum(1 for record in scan.records if record.extension in ANNOTATION_EXTENSIONS)

    profile = {
        "schema_version": SCHEMA_VERSION,
        "client_id": client_id,
        "dataset": _dataset_summary(root, data_card),
        "data_type": {
            "primary": _primary_data_type(image_records, tabular_count, annotation_count),
            "image_file_count": _safe_count(len(image_records), cfg.min_count),
            "primary_image_file_count": _safe_count(len(primary_image_records), cfg.min_count),
            "mask_image_file_count": _safe_count(len(mask_records), cfg.min_count),
            "tabular_file_count": _safe_count(tabular_count, cfg.min_count),
            "annotation_file_count": _safe_count(annotation_count, cfg.min_count),
            "archive_count": _safe_count(scan.archive_count, cfg.min_count),
        },
        "case_counts": {
            "train": _safe_count(split_counts["train"], cfg.min_count),
            "validation": _safe_count(split_counts["validation"], cfg.min_count),
            "test": _safe_count(split_counts["test"], cfg.min_count),
            "unknown": _safe_count(split_counts["unknown"], cfg.min_count),
            "total_primary_images": _safe_count(sum(split_counts.values()), cfg.min_count),
        },
        "image_dimensions": image_profile["dimensions"],
        "image_histogram": image_profile["histogram"],
        "labels": label_profile,
        "privacy": {
            "safe_to_share": True,
            "min_count": cfg.min_count,
            "redacted": [
                "local_file_paths",
                "filenames",
                "patient_ids",
                "per_sample_records",
                "raw_annotations",
                "raw_label_values",
                "raw_class_directory_names",
            ],
            "sampling": {
                "max_image_samples": cfg.max_image_samples,
                "histogram_bins": cfg.histogram_bins,
                "max_annotation_files": cfg.max_annotation_files,
                "max_annotation_rows": cfg.max_annotation_rows,
            },
        },
        "scan": {
            "local_file_count": _safe_count(scan.local_file_count, cfg.min_count),
            "archive_member_count": _safe_count(scan.archive_member_count, cfg.min_count),
            "profiled_record_count": _safe_count(len(scan.records), cfg.min_count),
            "scan_truncated": scan.scan_truncated,
            "client_identity_scope_applied": scan.client_identity_scope_applied,
        },
        "warnings": _dedupe(scan.warnings + image_profile["warnings"] + annotation_profile["warnings"]),
    }
    return profile


def parse_site_dataset(
    site_meta_path: str | Path,
    client_id: str,
    *,
    project_root: str | Path | None = None,
    config: DataParserConfig | None = None,
) -> dict[str, Any]:
    """Resolve a client's local data path from site metadata and profile it.

    The returned profile does not expose the resolved path. The server should
    use :func:`list_client_ids` instead of this function.
    """

    meta_path = Path(site_meta_path)
    site_meta = _load_json(meta_path)
    clients = _validated_site_clients(site_meta)
    client_id = validate_client_id(client_id)

    selected = None
    for client in clients:
        if isinstance(client, dict) and client.get("client_id") == client_id:
            selected = client
            break
    if selected is None:
        raise ValueError(f"client_id not found in site metadata: {client_id}")

    data_path_value = selected.get("data_path")
    if not isinstance(data_path_value, str) or not data_path_value:
        raise ValueError(f"client_id has no valid data_path: {client_id}")

    root = Path(project_root) if project_root is not None else meta_path.parent.parent
    data_path = Path(data_path_value)
    if not data_path.is_absolute():
        data_path = root / data_path

    profile = parse_dataset(data_path, client_id=client_id, config=config)
    profile["site_meta"] = {
        "client_id_resolved": client_id,
        "data_path_redacted": True,
    }
    return profile


def list_client_ids(site_meta_path: str | Path) -> dict[str, Any]:
    """Return the server-visible client list without local data paths."""

    site_meta = _load_json(Path(site_meta_path))
    clients = _validated_site_clients(site_meta)
    client_ids = [client["client_id"] for client in clients]
    return {
        "schema_version": SITE_CLIENTS_SCHEMA_VERSION,
        "client_count": len(client_ids),
        "clients": [{"client_id": client_id} for client_id in client_ids],
    }


def validate_client_id(client_id: str) -> str:
    """Validate an NVFlare client id before it becomes a path component."""

    if not isinstance(client_id, str) or not CLIENT_ID_PATTERN.fullmatch(client_id) or client_id in {".", ".."}:
        raise ValueError(
            "client_id must be 1-128 characters, start with an ASCII letter or digit, "
            "and contain only ASCII letters, digits, '.', '_', or '-'"
        )
    return client_id


def _validated_site_clients(site_meta: dict[str, Any]) -> list[dict[str, Any]]:
    clients = site_meta.get("clients")
    if not isinstance(clients, list):
        raise ValueError("site metadata must contain a clients list")

    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, client in enumerate(clients):
        if not isinstance(client, dict):
            raise ValueError(f"site metadata client entry {index} must be an object")
        try:
            client_id = validate_client_id(client.get("client_id"))
        except ValueError as exc:
            raise ValueError(f"site metadata client entry {index} has an invalid client_id: {exc}") from exc
        if client_id in seen:
            raise ValueError(f"site metadata contains duplicate client_id: {client_id}")
        seen.add(client_id)
        validated.append(client)
    return validated


def pseudonymize_site_metadata(
    site_meta_path: str | Path,
    *,
    output_path: str | Path,
    mapping_path: str | Path,
    prefix: str = "SITE",
) -> dict[str, Any]:
    """Create a runnable registry whose client ids do not name public datasets."""

    source = _load_json(Path(site_meta_path))
    clients = source.get("clients")
    if not isinstance(clients, list) or not clients:
        raise ValueError("site metadata must contain a non-empty clients list")
    clean_prefix = re.sub(r"[^A-Za-z0-9_]+", "_", prefix).strip("_").upper()
    if not clean_prefix:
        raise ValueError("pseudonym prefix must contain at least one letter or digit")
    width = max(3, len(str(len(clients))))
    transformed_clients: list[dict[str, Any]] = []
    mapping: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, client in enumerate(clients, start=1):
        if not isinstance(client, dict):
            raise ValueError(f"site metadata client entry {index - 1} must be an object")
        original_id = client.get("client_id")
        if not isinstance(original_id, str) or not original_id.strip():
            raise ValueError(f"site metadata client entry {index - 1} has no valid client_id")
        if original_id in seen:
            raise ValueError(f"site metadata contains duplicate client_id: {original_id}")
        seen.add(original_id)
        pseudonym = f"{clean_prefix}_{index:0{width}d}"
        transformed = dict(client)
        transformed["client_id"] = pseudonym
        transformed_clients.append(transformed)
        mapping.append({"pseudonym": pseudonym, "original_client_id": original_id})

    output_payload = {
        **{key: value for key, value in source.items() if key not in {"client_count", "clients"}},
        "schema_version": "fedready.pseudonymized_site_metadata.v1",
        "client_count": len(transformed_clients),
        "clients": transformed_clients,
        "identity_control": {
            "mode": "pseudonymized_client_ids",
            "prefix": clean_prefix,
            "mapping_stored_separately": True,
        },
    }
    mapping_payload = {
        "schema_version": "fedready.client_identity_mapping.v1",
        "privacy": {"safe_to_share": False, "reason": "links experiment aliases to source dataset ids"},
        "source_site_meta": str(Path(site_meta_path)),
        "pseudonymized_site_meta": str(Path(output_path)),
        "clients": mapping,
    }
    atomic_write_json(output_path, output_payload)
    atomic_write_json(mapping_path, mapping_payload)
    return {
        "schema_version": "fedready.pseudonymized_site_metadata_result.v1",
        "client_count": len(transformed_clients),
        "output_path": str(Path(output_path)),
        "mapping_path": str(Path(mapping_path)),
        "server_visible_client_ids": [entry["pseudonym"] for entry in mapping],
        "mapping_safe_to_share": False,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _load_data_card(data_path: Path) -> dict[str, Any] | None:
    candidates = []
    if data_path.is_dir():
        candidates.append(data_path / "data_card.json")
    else:
        candidates.append(data_path.with_name("data_card.json"))
    for candidate in candidates:
        if candidate.exists():
            try:
                return _load_json(candidate)
            except (OSError, ValueError, json.JSONDecodeError):
                return None
    return None


def load_shareable_label_schema(data_path: str | Path) -> dict[str, Any]:
    """Load only the task schema that a client data card explicitly authorizes."""

    card = _load_data_card(Path(data_path))
    return _data_card_shareable_label_schema(card)


def _scan_dataset(root: Path, config: DataParserConfig) -> ScanResult:
    warnings: list[str] = []
    records: list[FileRecord] = []
    archive_count = 0
    local_file_count = 0
    archive_member_count = 0
    scan_truncated = False

    def add_record(record: FileRecord) -> bool:
        nonlocal scan_truncated
        if len(records) >= config.max_scan_files:
            scan_truncated = True
            return False
        records.append(record)
        return True

    if not root.exists():
        return ScanResult(
            records=[],
            archive_count=0,
            local_file_count=0,
            archive_member_count=0,
            scan_truncated=False,
            warnings=["Dataset path is unavailable to the client runtime."],
        )

    local_files: Iterator[Path]
    if root.is_file():
        local_files = iter([root])
        base = root.parent
    else:
        local_files = (path for path in root.rglob("*") if path.is_file())
        base = root

    for file_path in local_files:
        local_file_count += 1
        archive_kind = _archive_kind(file_path)
        if config.include_archives and archive_kind:
            archive_count += 1
            archive_records, archive_warnings = _scan_archive(file_path, archive_kind)
            warnings.extend(archive_warnings)
            for archive_record in archive_records:
                archive_member_count += 1
                if not add_record(archive_record):
                    break
        else:
            try:
                logical_path = file_path.relative_to(base).as_posix()
            except ValueError:
                logical_path = file_path.name
            size = _safe_file_size(file_path)
            if not add_record(
                FileRecord(
                    storage="file",
                    container_path=file_path,
                    logical_path=logical_path,
                    extension=file_path.suffix.lower(),
                    size_bytes=size,
                )
            ):
                break
        if scan_truncated:
            warnings.append("File scan stopped after reaching the configured max_scan_files limit.")
            break

    return ScanResult(
        records=records,
        archive_count=archive_count,
        local_file_count=local_file_count,
        archive_member_count=archive_member_count,
        scan_truncated=scan_truncated,
        warnings=_dedupe(warnings),
    )


def _scope_scan_to_client_subdataset(scan: ScanResult, *, client_id: str | None) -> ScanResult:
    """Restrict a multi-dataset container to an unambiguous client-named subtree."""

    scope_key = _normalized_scope_component(client_id)
    if not scope_key or not scan.records:
        return scan
    scoped_records = [
        record
        for record in scan.records
        if any(_normalized_scope_component(part) == scope_key for part in PurePosixPath(record.logical_path).parts[:-1])
    ]
    if not scoped_records or len(scoped_records) == len(scan.records):
        return scan
    if not any(_is_image_record(record) for record in scoped_records):
        return scan
    return ScanResult(
        records=scoped_records,
        archive_count=scan.archive_count,
        local_file_count=scan.local_file_count,
        archive_member_count=scan.archive_member_count,
        scan_truncated=scan.scan_truncated,
        warnings=_dedupe(
            [
                *scan.warnings,
                "Client-local scan was scoped to a path component matching the configured client identity.",
            ]
        ),
        client_identity_scope_applied=True,
    )


def _normalized_scope_component(value: str | None) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _scan_archive(container_path: Path, archive_kind: str) -> tuple[list[FileRecord], list[str]]:
    if archive_kind == "zip":
        return _scan_zip(container_path)
    return _scan_tar(container_path)


def _scan_zip(container_path: Path) -> tuple[list[FileRecord], list[str]]:
    records: list[FileRecord] = []
    try:
        with zipfile.ZipFile(container_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                member = info.filename
                records.append(
                    FileRecord(
                        storage="zip",
                        container_path=container_path,
                        logical_path=f"{container_path.name}!/{member}",
                        extension=PurePosixPath(member).suffix.lower(),
                        size_bytes=info.file_size,
                        member_name=member,
                    )
                )
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return [], ["One archive could not be inspected."]
    return records, []


def _scan_tar(container_path: Path) -> tuple[list[FileRecord], list[str]]:
    records: list[FileRecord] = []
    try:
        with tarfile.open(container_path) as archive:
            for info in archive:
                if not info.isfile():
                    continue
                member = info.name
                records.append(
                    FileRecord(
                        storage="tar",
                        container_path=container_path,
                        logical_path=f"{container_path.name}!/{member}",
                        extension=PurePosixPath(member).suffix.lower(),
                        size_bytes=info.size,
                        member_name=member,
                    )
                )
    except (OSError, tarfile.TarError):
        return [], ["One archive could not be inspected."]
    return records, []


@contextmanager
def _open_record(record: FileRecord) -> Iterator[BinaryIO]:
    if record.storage == "file":
        with record.container_path.open("rb") as handle:
            yield handle
        return
    if record.storage == "zip":
        if record.member_name is None:
            raise ValueError("zip record missing member name")
        with zipfile.ZipFile(record.container_path) as archive:
            with archive.open(record.member_name) as handle:
                yield handle
        return
    if record.storage == "tar":
        if record.member_name is None:
            raise ValueError("tar record missing member name")
        with tarfile.open(record.container_path) as archive:
            handle = archive.extractfile(record.member_name)
            if handle is None:
                raise ValueError("tar member cannot be opened")
            with handle:
                yield handle
        return
    raise ValueError(f"unsupported storage kind: {record.storage}")


def _archive_kind(path: Path) -> str | None:
    name = path.name.lower()
    if name.endswith(".zip"):
        return "zip"
    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")):
        return "tar"
    return None


def _safe_file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _is_image_record(record: FileRecord) -> bool:
    return record.extension in IMAGE_EXTENSIONS


def _is_mask_record(record: FileRecord) -> bool:
    member = PurePosixPath(_member_only(record.logical_path))
    parts = [part for part in member.parts if part not in ("", ".")]
    if not parts:
        return False

    local_tokens = set(_tokens(member.stem))
    if len(parts) >= 2:
        local_tokens.update(_tokens(parts[-2]))

    strong_mask_hints = MASK_HINTS - {"seg", "segmentation"}
    if local_tokens & strong_mask_hints:
        return True

    # Treat bare immediate "seg"/"segmentation" roles as mask-like, but do not
    # let broad container names such as "seg_dataset" hide class-folder evidence.
    weak_hits = local_tokens & {"seg", "segmentation"}
    return bool(weak_hits and not (local_tokens & {"data", "dataset", "images", "image"}))


def _member_only(logical_path: str) -> str:
    marker = "!/"
    if marker in logical_path:
        return logical_path.split(marker, 1)[1]
    return logical_path


def _tokens(text: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]


def _safe_count(value: int, min_count: int) -> dict[str, Any]:
    if value == 0:
        return {"count": 0, "suppressed": False}
    if value < min_count:
        return {"count": None, "suppressed": True, "threshold": min_count}
    return {"count": value, "suppressed": False}


def _safe_counter(counter: Counter[str], min_count: int, *, limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, count in counter.most_common():
        if count < min_count:
            continue
        rows.append({"value": value, "count": count})
        if len(rows) >= limit:
            break
    return rows


def _safe_counter_counts(counter: Counter[str], min_count: int, *, limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (_value, count) in enumerate(counter.most_common(), start=1):
        if count < min_count:
            continue
        rows.append({"label_index": index, "count": count})
        if len(rows) >= limit:
            break
    return rows


def _dataset_summary(root: Path, data_card: dict[str, Any] | None) -> dict[str, Any]:
    dataset = data_card.get("dataset", {}) if data_card else {}
    if not isinstance(dataset, dict):
        dataset = {}
    fit_check = _first_mapping_value(data_card, ["fedagentbench_mapping", "rows", 0, "fit_check"])
    purpose = _clean_purpose(fit_check) if isinstance(fit_check, str) else None
    if purpose is None:
        purpose = _infer_purpose_from_text(" ".join(str(value) for value in dataset.values()))
    return {
        "display_name": _string_or_none(dataset.get("display_name")) or _display_name_from_path(root),
        "purpose": purpose,
        "domain": _string_or_none(dataset.get("domain")),
        "source_type": _string_or_none(dataset.get("source_type")),
        "kaggle_ref": _string_or_none(dataset.get("kaggle_ref")),
        "kaggle_url": _string_or_none(dataset.get("kaggle_url")),
        "license": _string_or_none(dataset.get("kaggle_license")),
    }


def _first_mapping_value(container: Any, path: list[Any]) -> Any:
    value = container
    for part in path:
        if isinstance(part, int):
            if not isinstance(value, list) or len(value) <= part:
                return None
            value = value[part]
        else:
            if not isinstance(value, dict):
                return None
            value = value.get(part)
    return value


def _clean_purpose(text: str) -> str:
    cleaned = re.sub(r"^\s*(fits|partial|does not fit)\s*:\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:/workspace|FedReady|data/dataset_[^\s,;]+)[^\s,;]*", "[redacted_path]", cleaned)
    return cleaned.strip()


def _infer_purpose_from_text(text: str) -> str | None:
    lowered = text.lower()
    terms = _label_terms_from_text(lowered)
    if "seg" in lowered and terms:
        return f"{', '.join(terms[:5])} segmentation"
    if "classification" in lowered or "diagnosis" in lowered:
        if terms:
            return f"{', '.join(terms[:5])} classification"
        return "image classification or diagnosis"
    if terms:
        return f"image task involving {', '.join(terms[:5])}"
    return None


def _display_name_from_path(path: Path) -> str:
    name = path.stem if path.is_file() else path.name
    name = re.sub(r"[_-]+", " ", name).strip()
    return name.title() if name else "Local Dataset"


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _primary_data_type(
    image_records: list[FileRecord],
    tabular_count: int,
    annotation_count: int,
) -> str:
    if image_records:
        return "image"
    if tabular_count:
        return "tabular"
    if annotation_count:
        return "annotation"
    return "unknown"


def _count_splits(records: list[FileRecord]) -> Counter[str]:
    counts: Counter[str] = Counter({"train": 0, "validation": 0, "test": 0, "unknown": 0})
    for record in records:
        split = _infer_split(record.logical_path)
        counts[split] += 1
    return counts


def _infer_split(logical_path: str) -> str:
    for token in _tokens(_member_only(logical_path)):
        if token in SPLIT_ALIASES:
            return SPLIT_ALIASES[token]
    return "unknown"


def _infer_class_labels(records: list[FileRecord], config: DataParserConfig) -> Counter[str]:
    labels: Counter[str] = Counter()
    for record in records:
        if _is_mask_record(record):
            continue
        label = _class_label_from_path(record.logical_path)
        if label is not None:
            labels[label] += 1
    return Counter({label: count for label, count in labels.items() if count >= config.min_count})


def _class_label_from_path(logical_path: str) -> str | None:
    parts = [part for part in PurePosixPath(_member_only(logical_path)).parts if part not in ("", ".")]
    lowered = [part.lower() for part in parts]
    for index, part in enumerate(lowered[:-1]):
        split = SPLIT_ALIASES.get(_sanitize_label(part))
        if split and index + 1 < len(parts) - 1:
            return _valid_class_label(parts[index + 1])
    if len(parts) >= 2:
        return _valid_class_label(parts[-2])
    return None


def _valid_class_label(value: str) -> str | None:
    label = _sanitize_label(value)
    if not label or label in GENERIC_LABEL_TOKENS:
        return None
    if label.isdigit() or len(label) > 64:
        return None
    if re.search(r"\d{4,}", label):
        return None
    return label


def _sanitize_label(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _profile_annotations(records: list[FileRecord], config: DataParserConfig) -> dict[str, Any]:
    annotation_records = [record for record in records if record.extension in ANNOTATION_EXTENSIONS]
    csv_records = [record for record in annotation_records if record.extension in CSV_EXTENSIONS]
    label_columns: Counter[str] = Counter()
    label_values: Counter[str] = Counter()
    warnings: list[str] = []
    parsed_files = 0
    parsed_rows = 0

    for record in csv_records[: config.max_annotation_files]:
        try:
            raw_text = _read_record_text(record, config.max_annotation_bytes)
        except OSError:
            warnings.append("One annotation file could not be read.")
            continue
        delimiter = "\t" if record.extension == ".tsv" else ","
        reader = csv.DictReader(io.StringIO(raw_text), delimiter=delimiter)
        if not reader.fieldnames:
            continue
        candidate_columns = [
            column
            for column in reader.fieldnames
            if _looks_like_label_column(column) and not _looks_like_sensitive_annotation_column(column)
        ]
        for column in candidate_columns:
            sanitized = _sanitize_label(column)
            if sanitized:
                label_columns[sanitized] += 1
        row_budget = max(config.max_annotation_rows - parsed_rows, 0)
        if row_budget == 0:
            break
        for index, row in enumerate(reader):
            if index >= row_budget:
                break
            parsed_rows += 1
            for column in candidate_columns:
                raw_value = row.get(column)
                if raw_value is None:
                    continue
                for label in _label_values_from_cell(raw_value):
                    label_values[label] += 1
        parsed_files += 1

    return {
        "annotation_file_count": len(annotation_records),
        "parsed_tabular_annotation_files": parsed_files,
        "parsed_annotation_rows": parsed_rows,
        "label_columns": Counter({label: count for label, count in label_columns.items() if count >= config.min_count}),
        "label_values": Counter({label: count for label, count in label_values.items() if count >= config.min_count}),
        "label_terms": _record_label_terms(annotation_records, min_count=config.min_count),
        "has_segmentation_hint": _has_segmentation_annotation_hint(annotation_records),
        "warnings": _dedupe(warnings),
    }


def _read_record_text(record: FileRecord, max_bytes: int) -> str:
    with _open_record(record) as handle:
        data = handle.read(max_bytes)
    return data.decode("utf-8", errors="replace")


def _looks_like_label_column(column: str) -> bool:
    normalized = _sanitize_label(column)
    if normalized in LABEL_COLUMN_EXACT_NAMES:
        return True
    tokens = set(_tokens(column))
    if not tokens or _looks_like_sensitive_annotation_column(column):
        return False
    allowed_tokens = LABEL_COLUMN_CORE_TOKENS | LABEL_COLUMN_SAFE_MODIFIER_TOKENS
    return bool(tokens & LABEL_COLUMN_CORE_TOKENS) and tokens <= allowed_tokens


def _looks_like_sensitive_annotation_column(column: str) -> bool:
    normalized = _sanitize_label(column)
    if normalized in LABEL_COLUMN_EXACT_NAMES:
        return False
    tokens = set(_tokens(column)) | {normalized}
    return bool(tokens & SENSITIVE_ANNOTATION_COLUMN_TOKENS)


def _label_values_from_cell(value: str) -> list[str]:
    cleaned = value.strip()
    if not cleaned:
        return []
    if len(cleaned) > 80 or _looks_like_sensitive_label_value(cleaned):
        return []
    values = re.split(r"[;|,]", cleaned)
    labels: list[str] = []
    for item in values:
        if _looks_like_sensitive_label_value(item):
            continue
        label = _valid_class_label(item)
        if label and label not in {"0", "1", "true", "false", "yes", "no"}:
            labels.append(label)
    return labels


def _looks_like_sensitive_label_value(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    lowered = text.lower()
    if "@" in text or "http://" in lowered or "https://" in lowered:
        return True
    tokens = re.findall(r"[A-Za-z]+", text)
    normalized_tokens = {_sanitize_label(token) for token in tokens}
    org_tokens = {
        "clinic",
        "company",
        "corp",
        "department",
        "hospital",
        "inc",
        "institute",
        "institution",
        "laboratory",
        "lab",
        "llc",
        "ltd",
        "medical",
        "organization",
        "university",
    }
    if normalized_tokens & org_tokens:
        return True
    if not (2 <= len(tokens) <= 4):
        return False
    if normalized_tokens & LABEL_VALUE_DOMAIN_TOKENS:
        return False
    alpha_text = re.sub(r"[^A-Za-z]+", "", text)
    if not alpha_text:
        return False
    title_like = all(token[:1].isupper() and token[1:].islower() for token in tokens if len(token) > 1)
    all_words = all(token.isalpha() for token in tokens)
    return all_words and (title_like or text == text.lower())


def _profile_labels(
    data_card: dict[str, Any] | None,
    mask_records: list[FileRecord],
    class_labels: Counter[str],
    annotation_profile: dict[str, Any],
    image_records: list[FileRecord],
    config: DataParserConfig,
) -> dict[str, Any]:
    sources: list[str] = []
    meanings: list[str] = []
    share_label_vocabulary = _data_card_allows_label_vocabulary_sharing(data_card)
    shareable_schema = _data_card_shareable_label_schema(data_card)
    shareable_concepts = shareable_schema["shareable_concepts"]
    shareable_value_meanings = shareable_schema["shareable_value_meanings"]

    context_text = _data_card_label_context(data_card)
    meanings.extend(_label_terms_from_text(context_text))
    meanings.extend(shareable_concepts)
    meanings.extend(shareable_value_meanings.values())

    if mask_records:
        sources.append("mask_image_paths")
        if share_label_vocabulary:
            meanings.extend(_record_label_terms(mask_records, min_count=config.min_count))

    if class_labels:
        sources.append("class_directories")
        if share_label_vocabulary:
            meanings.extend(class_labels.keys())

    label_columns = annotation_profile["label_columns"]
    label_values = annotation_profile["label_values"]
    annotation_meanings = annotation_profile["label_terms"]
    if label_columns or label_values:
        sources.append("tabular_annotations")
        meanings.extend(label_columns.keys())
        if share_label_vocabulary:
            meanings.extend(label_values.keys())
    if annotation_profile["annotation_file_count"]:
        if share_label_vocabulary:
            meanings.extend(annotation_meanings)
        if annotation_profile["has_segmentation_hint"]:
            sources.append("structured_segmentation_annotations")

    label_type = "unknown"
    if mask_records:
        label_type = "segmentation_mask"
    elif _has_segmentation_channel_labels(class_labels, context_text, image_records, config.min_count):
        label_type = "segmentation_channel"
        sources.append("segmentation_channel_directories")
    elif _has_structured_segmentation_annotations(annotation_profile, context_text):
        label_type = "contour_or_structured_segmentation_annotation"
    elif label_values or class_labels:
        label_type = "classification_or_grading"
    elif annotation_profile["annotation_file_count"]:
        label_type = "structured_annotation"

    return {
        "label_type": label_type,
        "label_meanings": sorted(_dedupe(meanings))[:40],
        "declared_shareable_concepts": shareable_concepts,
        "declared_shareable_value_meanings": shareable_value_meanings,
        "label_source": sorted(_dedupe(sources)),
        "label_vocabulary_shareable": share_label_vocabulary,
        "label_vocabulary_privacy": {
            "raw_values_redacted": not share_label_vocabulary,
            "shareable_declared_by_data_card": (
                share_label_vocabulary or bool(shareable_concepts) or bool(shareable_value_meanings)
            ),
            "declared_concepts_shareable": bool(shareable_concepts),
            "declared_value_meanings_shareable": bool(shareable_value_meanings),
        },
        "mask_image_count": _safe_count(len(mask_records), config.min_count),
        "class_labels": _safe_counter(class_labels, config.min_count) if share_label_vocabulary else [],
        "class_label_cardinality": len(class_labels),
        "class_label_counts": _safe_counter_counts(class_labels, config.min_count),
        "annotation_file_count": _safe_count(annotation_profile["annotation_file_count"], config.min_count),
        "annotation_label_columns": _safe_counter(label_columns, config.min_count),
        "annotation_label_values": _safe_counter(label_values, config.min_count) if share_label_vocabulary else [],
        "annotation_label_value_cardinality": len(label_values),
        "annotation_label_value_counts": _safe_counter_counts(label_values, config.min_count),
        "annotation_label_meanings": sorted(_dedupe(annotation_meanings))[:40] if share_label_vocabulary else [],
        "annotation_has_segmentation_hint": bool(annotation_profile["has_segmentation_hint"]),
    }


def _label_terms_from_text(text: str, *, limit: int = 40) -> list[str]:
    """Extract bounded observed terms without task/domain-specific normalization."""

    terms: list[str] = []
    for raw_token in _tokens(text):
        token = _valid_label_term(raw_token)
        if token:
            terms.append(token)
    return _dedupe(terms)[:limit]


def _data_card_label_context(data_card: dict[str, Any] | None) -> str:
    if not data_card:
        return ""
    fragments: list[str] = []
    dataset = data_card.get("dataset")
    if isinstance(dataset, dict):
        for key in ("display_name", "description"):
            value = dataset.get(key)
            if isinstance(value, str):
                fragments.append(value)
    fit_check = _first_mapping_value(data_card, ["fedagentbench_mapping", "rows", 0, "fit_check"])
    if isinstance(fit_check, str):
        fragments.append(_clean_purpose(fit_check))
    return " ".join(fragments)


def _data_card_allows_label_vocabulary_sharing(data_card: dict[str, Any] | None) -> bool:
    if not isinstance(data_card, dict):
        return False
    return _first_mapping_value(data_card, ["privacy", "label_vocabulary_shareable"]) is True


def _data_card_shareable_label_schema(data_card: dict[str, Any] | None) -> dict[str, Any]:
    """Return only client-authored task schema, never observed label values."""

    if not isinstance(data_card, dict):
        return {"shareable_concepts": [], "shareable_value_meanings": {}}
    values = _first_mapping_value(data_card, ["label_schema", "shareable_concepts"])
    concepts: list[str] = []
    if isinstance(values, list):
        for value in values[:40]:
            if not isinstance(value, str):
                continue
            concept = _valid_label_term(value)
            if concept:
                concepts.append(concept)

    raw_meanings = _first_mapping_value(data_card, ["label_schema", "shareable_value_meanings"])
    value_meanings: dict[str, str] = {}
    if isinstance(raw_meanings, dict):
        for raw_value, raw_meaning in list(raw_meanings.items())[:40]:
            if not isinstance(raw_value, str) or not isinstance(raw_meaning, str):
                continue
            value = raw_value.strip()
            meaning = _valid_label_term(raw_meaning)
            if value and len(value) <= 64 and not any(char in value for char in "\r\n\t") and meaning:
                value_meanings[value] = meaning
    return {
        "shareable_concepts": sorted(_dedupe(concepts))[:40],
        "shareable_value_meanings": dict(sorted(value_meanings.items())),
    }


def _record_label_terms(records: list[FileRecord], *, min_count: int, limit: int = 40) -> list[str]:
    """Extract aggregate label-like terms from directory/container names only."""

    counts: Counter[str] = Counter()
    for record in records[:1000]:
        member = PurePosixPath(_member_only(record.logical_path))
        record_terms: set[str] = set()
        for part in member.parts[:-1]:
            record_terms.update(_label_terms_from_text(part, limit=limit))
        counts.update(record_terms)
    return sorted(term for term, count in counts.items() if count >= min_count)[:limit]


def _valid_label_term(value: str) -> str | None:
    label = _sanitize_label(value)
    if not label or label in GENERIC_LABEL_TOKENS:
        return None
    if len(label) <= 1 or len(label) > 64:
        return None
    if label.isdigit() or re.search(r"\d{4,}", label):
        return None
    return label


def _has_segmentation_annotation_hint(annotation_records: list[FileRecord]) -> bool:
    for record in annotation_records[:1000]:
        tokens = set(_tokens(_member_only(record.logical_path)))
        if tokens & SEGMENTATION_ANNOTATION_HINTS:
            return True
    return False


def _has_segmentation_channel_labels(
    class_labels: Counter[str],
    context_text: str,
    image_records: list[FileRecord],
    min_count: int,
) -> bool:
    if not class_labels:
        return False
    lowered = context_text.lower()
    return (
        "segmentation channel" in lowered
        or "segmentation channels" in lowered
        or ("multi-channel" in lowered and "segmentation" in lowered)
        or _has_parallel_image_channels(image_records, class_labels, min_count=min_count)
    )


def _has_parallel_image_channels(
    records: list[FileRecord],
    class_labels: Counter[str],
    *,
    min_count: int,
) -> bool:
    source_roles = {"image", "images", "input", "inputs", "original", "raw", "scan", "source", "volume"}
    source_stems: dict[str, set[str]] = {}
    channel_stems: dict[tuple[str, str], set[str]] = {}
    for record in records[:5000]:
        member = PurePosixPath(_member_only(record.logical_path))
        if len(member.parts) < 2:
            continue
        role = _sanitize_label(member.parent.name)
        cohort = member.parent.parent.as_posix().lower()
        stem = member.stem.lower()
        if role in source_roles:
            source_stems.setdefault(cohort, set()).add(stem)
        elif role in class_labels:
            channel_stems.setdefault((cohort, role), set()).add(stem)
    required_matches = max(2, min_count)
    for (cohort, _role), stems in channel_stems.items():
        matching_source_stems = source_stems.get(cohort, set())
        if len(stems & matching_source_stems) >= required_matches:
            return True
    return False


def _has_structured_segmentation_annotations(
    annotation_profile: dict[str, Any],
    context_text: str,
) -> bool:
    if not annotation_profile["annotation_file_count"]:
        return False
    lowered = context_text.lower()
    return bool(annotation_profile["has_segmentation_hint"]) or "segmentation" in lowered


def _profile_images(records: list[FileRecord], config: DataParserConfig) -> dict[str, Any]:
    warnings: list[str] = []
    if Image is None:
        unavailable = {
            "available": False,
            "reason": "pillow_unavailable",
            "sampled_images": _safe_count(0, config.min_count),
        }
        return {"dimensions": unavailable, "histogram": unavailable, "warnings": warnings}

    sample_records = records[: config.max_image_samples]
    widths: list[int] = []
    heights: list[int] = []
    channels: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    histogram = [0] * 256
    histogram_pixel_count = 0
    failures = 0

    resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)

    for record in sample_records:
        try:
            with _open_record(record) as handle:
                with Image.open(handle) as image:
                    width, height = image.size
                    widths.append(int(width))
                    heights.append(int(height))
                    channel_count = len(image.getbands())
                    channels[str(channel_count)] += 1
                    sizes[f"{width}x{height}"] += 1

                    grayscale = image.convert("L")
                    grayscale.thumbnail((config.histogram_max_side, config.histogram_max_side), resampling)
                    image_histogram = grayscale.histogram()
                    histogram = [left + right for left, right in zip(histogram, image_histogram)]
                    histogram_pixel_count += grayscale.width * grayscale.height
        except Exception:  # noqa: BLE001 - third-party image loaders raise many exception types.
            failures += 1

    measured = len(widths)
    if failures:
        warnings.append("Some image samples could not be decoded.")

    if measured < config.min_count:
        dimensions = {
            "available": False,
            "reason": "insufficient_decodable_samples",
            "sampled_images": _safe_count(measured, config.min_count),
        }
        histogram_profile = {
            "available": False,
            "reason": "insufficient_decodable_samples",
            "sampled_images": _safe_count(measured, config.min_count),
        }
    else:
        dimensions = {
            "available": True,
            "sampled_images": _safe_count(measured, config.min_count),
            "width": _numeric_summary(widths),
            "height": _numeric_summary(heights),
            "channels": _safe_counter(channels, config.min_count),
            "common_sizes": [
                _size_row(size, count) for size, count in sizes.most_common(20) if count >= config.min_count
            ],
        }
        histogram_profile = {
            "available": histogram_pixel_count > 0,
            "sampled_images": _safe_count(measured, config.min_count),
            "bins": _histogram_bins(histogram, config.histogram_bins),
        }

    return {"dimensions": dimensions, "histogram": histogram_profile, "warnings": warnings}


def _numeric_summary(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": _percentile(ordered, 0.10),
        "median": _percentile(ordered, 0.50),
        "mean": round(sum(ordered) / len(ordered), 2),
        "p90": _percentile(ordered, 0.90),
        "max": ordered[-1],
    }


def _percentile(ordered_values: list[int], q: float) -> float:
    if len(ordered_values) == 1:
        return float(ordered_values[0])
    position = (len(ordered_values) - 1) * q
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered_values) - 1)
    fraction = position - lower_index
    lower = ordered_values[lower_index]
    upper = ordered_values[upper_index]
    return round(lower + (upper - lower) * fraction, 2)


def _size_row(size: str, count: int) -> dict[str, int]:
    width, height = size.split("x", 1)
    return {"width": int(width), "height": int(height), "count": count}


def _histogram_bins(histogram: list[int], bins: int) -> list[dict[str, Any]]:
    total = sum(histogram)
    if total == 0:
        return []
    rows = []
    for index in range(bins):
        start = round(index * 256 / bins)
        stop = round((index + 1) * 256 / bins) - 1
        count = sum(histogram[start : stop + 1])
        rows.append(
            {
                "range": [start, stop],
                "proportion": round(count / total, 6),
            }
        )
    return rows


def _dedupe(values: list[str] | tuple[str, ...]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile a local FedReady client dataset.")
    parser.add_argument("data_path", nargs="?", help="Local dataset path to profile.")
    parser.add_argument("--site-meta", help="Resolve data_path from a site-meta.json file.")
    parser.add_argument("--client-id", help="Client id to use for the profile.")
    parser.add_argument("--project-root", help="Project root for relative site metadata paths.")
    parser.add_argument("--min-count", type=int, default=DataParserConfig.min_count)
    parser.add_argument("--max-image-samples", type=int, default=DataParserConfig.max_image_samples)
    parser.add_argument("--histogram-bins", type=int, default=DataParserConfig.histogram_bins)
    args = parser.parse_args(argv)

    config = DataParserConfig(
        min_count=args.min_count,
        max_image_samples=args.max_image_samples,
        histogram_bins=args.histogram_bins,
    )
    if args.site_meta:
        if not args.client_id:
            parser.error("--client-id is required with --site-meta")
        profile = parse_site_dataset(
            args.site_meta,
            args.client_id,
            project_root=args.project_root,
            config=config,
        )
    else:
        if not args.data_path:
            parser.error("data_path is required unless --site-meta is used")
        profile = parse_dataset(args.data_path, client_id=args.client_id, config=config)
    print(json.dumps(profile, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
