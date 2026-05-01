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

import io
import zipfile
from zipfile import ZipFile

import pytest

from nvflare.apis.utils.job_submit_token import canonical_job_content_hash
from nvflare.lighter.tool_consts import NVFLARE_SIG_FILE, NVFLARE_SUBMITTER_CRT_FILE


def _zip_bytes(files):
    output = io.BytesIO()
    with ZipFile(output, "w") as zip_file:
        for name, content in files.items():
            zip_file.writestr(name, content)
    return output.getvalue()


def test_canonical_job_content_hash_ignores_signing_artifacts():
    base = _zip_bytes(
        {
            "hello/meta.json": "{}",
            "hello/app/config/config_fed_server.json": "{}",
        }
    )
    signed = _zip_bytes(
        {
            "hello/meta.json": "{}",
            "hello/app/config/config_fed_server.json": "{}",
            f"hello/{NVFLARE_SIG_FILE}": '{"signature": "volatile"}',
            f"hello/{NVFLARE_SUBMITTER_CRT_FILE}": "volatile cert",
        }
    )

    assert canonical_job_content_hash(base) == canonical_job_content_hash(signed)


def test_canonical_job_content_hash_skips_directory_symlinks(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "meta.json").write_text("{}")
    link = job_dir / "linked-outside.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("filesystem does not support symlinks")

    without_link = tmp_path / "job-without-link"
    without_link.mkdir()
    (without_link / "meta.json").write_text("{}")

    assert canonical_job_content_hash(str(job_dir)) == canonical_job_content_hash(str(without_link))


def test_canonical_job_content_hash_matches_directory_and_wrapped_zip(tmp_path):
    job_dir = tmp_path / "hello"
    app_dir = job_dir / "app" / "config"
    app_dir.mkdir(parents=True)
    (job_dir / "meta.json").write_text("{}", encoding="utf-8")
    (app_dir / "config_fed_server.json").write_text("{}", encoding="utf-8")
    wrapped_zip = _zip_bytes(
        {
            "hello/meta.json": "{}",
            "hello/app/config/config_fed_server.json": "{}",
        }
    )

    assert canonical_job_content_hash(str(job_dir)) == canonical_job_content_hash(wrapped_zip)


def test_canonical_job_content_hash_treats_parent_directory_as_different_content(tmp_path):
    job_dir = tmp_path / "hello"
    job_dir.mkdir()
    (job_dir / "meta.json").write_text("{}", encoding="utf-8")

    assert canonical_job_content_hash(str(tmp_path)) != canonical_job_content_hash(str(job_dir))


def test_canonical_job_content_hash_rejects_dot_zip_member():
    with pytest.raises(ValueError, match="unsafe path"):
        canonical_job_content_hash(_zip_bytes({".": "not a valid job member"}))


def test_canonical_job_content_hash_rejects_oversized_zip_member(monkeypatch):
    import nvflare.apis.utils.job_submit_token as job_submit_token

    monkeypatch.setattr(job_submit_token, "_MAX_HASH_ZIP_MEMBER_SIZE", 16)
    output = io.BytesIO()
    with ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("hello/meta.json", "x" * 17)

    with pytest.raises(ValueError, match="zip member exceeds size limit"):
        canonical_job_content_hash(output.getvalue())
