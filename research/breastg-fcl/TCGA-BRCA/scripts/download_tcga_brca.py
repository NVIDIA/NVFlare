#!/usr/bin/env python3
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

# The original BreastG-FCL MIT notice is retained below for the upstream code.
# MIT License
#
# Copyright (c) 2026 IntelliSys-Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Download public TCGA-BRCA STAR count files from the NCI GDC API."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import requests

GDC_API = "https://api.gdc.cancer.gov"
PROJECT_ID = "TCGA-BRCA"
FIELDS = [
    "file_id",
    "file_name",
    "file_size",
    "md5sum",
    "access",
    "data_category",
    "data_type",
    "cases.submitter_id",
    "cases.case_id",
    "cases.samples.submitter_id",
    "cases.samples.sample_type",
    "cases.samples.tissue_type",
    "analysis.workflow_type",
]


def build_filters() -> dict:
    return {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [PROJECT_ID],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_category",
                    "value": ["Transcriptome Profiling"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_type",
                    "value": ["Gene Expression Quantification"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.analysis.workflow_type",
                    "value": ["STAR - Counts"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.access",
                    "value": ["open"],
                },
            },
        ],
    }


def request_json(session: requests.Session, url: str, **kwargs) -> dict:
    response = session.get(url, timeout=90, **kwargs)
    response.raise_for_status()
    return response.json()


def fetch_status(session: requests.Session) -> dict:
    return request_json(session, f"{GDC_API}/status")


def fetch_manifest(session: requests.Session, page_size: int = 1000) -> list[dict]:
    filters = build_filters()
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(FIELDS),
        "format": "JSON",
        "size": page_size,
        "from": 0,
        "sort": "file_id:asc",
    }

    files: list[dict] = []
    while True:
        data = request_json(session, f"{GDC_API}/files", params=params)["data"]
        files.extend(data["hits"])
        pagination = data["pagination"]
        if len(files) >= pagination["total"]:
            break
        params["from"] += page_size
    return files


def first_nested(hit: dict, path: list[str], default: str = "") -> str:
    value = hit
    for key in path:
        if isinstance(value, list):
            value = value[0] if value else {}
        if not isinstance(value, dict):
            return default
        value = value.get(key, {})
    if isinstance(value, list):
        value = value[0] if value else default
    return str(value) if value not in ({}, None) else default


def write_manifest(files: list[dict], path: Path) -> None:
    columns = [
        "file_id",
        "file_name",
        "file_size",
        "md5sum",
        "case_submitter_id",
        "case_id",
        "sample_submitter_id",
        "sample_type",
        "tissue_type",
        "workflow_type",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for hit in files:
            row = {
                "file_id": hit.get("file_id", hit.get("id", "")),
                "file_name": hit.get("file_name", ""),
                "file_size": str(hit.get("file_size", "")),
                "md5sum": hit.get("md5sum", ""),
                "case_submitter_id": first_nested(hit, ["cases", "submitter_id"]),
                "case_id": first_nested(hit, ["cases", "case_id"]),
                "sample_submitter_id": first_nested(hit, ["cases", "samples", "submitter_id"]),
                "sample_type": first_nested(hit, ["cases", "samples", "sample_type"]),
                "tissue_type": first_nested(hit, ["cases", "samples", "tissue_type"]),
                "workflow_type": first_nested(hit, ["analysis", "workflow_type"]),
            }
            handle.write("\t".join(row[column] for column in columns) + "\n")


def chunked(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("chunk size must be a positive integer")
    return [items[index : index + size] for index in range(0, len(items), size)]


def download_target(raw_dir: Path, file_id: str, filename: str) -> Path:
    """Keep manifest paths and existing symlinks within the download directory."""
    for field, value in (("file_id", file_id), ("file_name", filename)):
        if not isinstance(value, str) or value in ("", ".", "..") or "/" in value or "\\" in value:
            raise ValueError(f"Invalid manifest {field}: expected a single path component")
    root = raw_dir.resolve()
    target = (root / file_id / filename).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Manifest target escapes the download directory: {file_id}/{filename}")
    return target


def file_matches_manifest(hit: dict, raw_dir: Path) -> bool:
    """Check the actual expression file; a download ledger is not evidence."""
    file_id = hit.get("file_id", hit.get("id"))
    filename = hit.get("file_name")
    if not file_id or not filename:
        return False
    path = download_target(raw_dir, file_id, filename)
    if not path.is_file() or path.stat().st_size != int(hit["file_size"]):
        return False
    expected_md5 = hit.get("md5sum")
    if expected_md5:
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_md5.lower():
            return False
    return True


def save_downloaded(path: Path, downloaded: set[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sorted(downloaded), handle, indent=2)


def extract_download_tar(tar_path: Path, raw_dir: Path, expected_files: dict[str, str]) -> list[str]:
    """Stream only requested GDC files, without applying archive paths or metadata."""
    extracted_ids: list[str] = []
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported archive member: {member.name}")
            parts = member.name.split("/")
            if "\\" in member.name or any(part in ("", ".", "..") for part in parts):
                raise ValueError(f"Unsafe archive member path: {member.name}")
            # GDC also includes top-level metadata such as MANIFEST.txt.
            if len(parts) == 1:
                continue
            if len(parts) != 2:
                raise ValueError(f"Unexpected archive member path: {member.name}")
            file_id, filename = parts
            if expected_files.get(file_id) != filename:
                raise ValueError(f"Archive member is not in the requested manifest: {member.name}")
            target = download_target(raw_dir, file_id, filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted_ids.append(file_id)
    return extracted_ids


def download_files(
    session: requests.Session,
    files: list[dict],
    raw_dir: Path,
    metadata_dir: Path,
    chunk_size: int,
    pause_seconds: float,
) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    downloaded_path = metadata_dir / "downloaded_files.json"
    file_ids = [hit.get("file_id", hit.get("id")) for hit in files]
    files_by_id = {hit.get("file_id", hit.get("id")): hit for hit in files}
    downloaded = {file_id for file_id in file_ids if file_id and file_matches_manifest(files_by_id[file_id], raw_dir)}
    # Reconcile stale/missing ledgers with disk before attempting any download.
    save_downloaded(downloaded_path, downloaded)
    pending = [file_id for file_id in file_ids if file_id and file_id not in downloaded]

    for index, ids in enumerate(chunked(pending, chunk_size), start=1):
        print(f"Downloading chunk {index}: {len(ids)} files")
        response = session.post(
            f"{GDC_API}/data",
            json={"ids": ids},
            timeout=600,
            stream=True,
        )
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(prefix="gdc_", suffix=".tar.gz", delete=False) as handle:
            tmp_path = Path(handle.name)
        try:
            with tmp_path.open("wb") as handle:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    if block:
                        handle.write(block)
            extract_download_tar(tmp_path, raw_dir, {file_id: files_by_id[file_id]["file_name"] for file_id in ids})
        finally:
            tmp_path.unlink(missing_ok=True)

        verified = {file_id for file_id in ids if file_matches_manifest(files_by_id[file_id], raw_dir)}
        downloaded.update(verified)
        save_downloaded(downloaded_path, downloaded)
        failed = [file_id for file_id in ids if file_id not in verified]
        if failed:
            raise RuntimeError(f"Downloaded files failed manifest validation: {', '.join(failed)}")
        print(f"Downloaded {len(downloaded)} / {len(file_ids)} files")
        if pause_seconds:
            time.sleep(pause_seconds)


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_dir)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be a positive integer")
    return args


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    metadata_dir = root / "metadata"
    raw_dir = root / "data" / "raw"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    status = fetch_status(session)
    files = fetch_manifest(session)

    (metadata_dir / "gdc_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (metadata_dir / "gdc_query.json").write_text(
        json.dumps(
            {
                "api": GDC_API,
                "project_id": PROJECT_ID,
                "filters": build_filters(),
                "fields": FIELDS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_manifest(files, metadata_dir / "gdc_files_manifest.tsv")

    total_size = sum(int(hit.get("file_size", 0)) for hit in files)
    print(f"Found {len(files)} files ({total_size / 1024**3:.2f} GiB raw).")
    print(f"GDC data release: {status.get('data_release', 'unknown')}")

    if args.metadata_only:
        return

    download_files(
        session=session,
        files=files,
        raw_dir=raw_dir,
        metadata_dir=metadata_dir,
        chunk_size=args.chunk_size,
        pause_seconds=args.pause_seconds,
    )


if __name__ == "__main__":
    main()
