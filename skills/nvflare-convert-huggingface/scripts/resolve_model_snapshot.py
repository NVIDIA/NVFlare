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

"""Resolve a local or Hugging Face model snapshot without false failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def _load_huggingface_hub():
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    return snapshot_download, LocalEntryNotFoundError


def _looks_like_local_path(identifier: str) -> bool:
    return Path(identifier).is_absolute() or identifier.startswith(("./", "../", "~"))


def resolve_model_snapshot(
    identifier: str,
    *,
    allow_download: bool = False,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Return structured resolution evidence; an expected cache miss is not an error."""
    candidate = Path(identifier).expanduser()
    if candidate.exists():
        return {
            "download_authorized": allow_download,
            "identifier": identifier,
            "resolved_path": str(candidate.resolve()),
            "source": "local",
            "status": "available",
        }
    if _looks_like_local_path(identifier):
        return {
            "download_authorized": allow_download,
            "identifier": identifier,
            "reason": "local_path_not_found",
            "resolved_path": None,
            "source": "local",
            "status": "missing",
        }

    snapshot_download, local_entry_not_found = _load_huggingface_hub()
    kwargs: Dict[str, Any] = {
        "local_files_only": not allow_download,
        "repo_id": identifier,
    }
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir

    try:
        resolved_path = snapshot_download(**kwargs)
    except local_entry_not_found:
        if allow_download:
            raise
        return {
            "download_authorized": False,
            "identifier": identifier,
            "reason": "not_cached",
            "resolved_path": None,
            "source": "hub_cache",
            "status": "missing",
        }

    return {
        "download_authorized": allow_download,
        "identifier": identifier,
        "resolved_path": str(Path(resolved_path).resolve()),
        "source": "hub_download_or_cache" if allow_download else "hub_cache",
        "status": "available",
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("identifier", help="Local model path or Hugging Face model repository ID")
    parser.add_argument("--allow-download", action="store_true", help="Permit one normal cache-aware Hub download")
    parser.add_argument("--revision", help="Optional Hub revision")
    parser.add_argument("--cache-dir", help="Optional Hugging Face cache directory")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    result = resolve_model_snapshot(
        args.identifier,
        allow_download=args.allow_download,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
