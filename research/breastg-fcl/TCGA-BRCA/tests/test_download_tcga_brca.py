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

"""Offline coverage of GDC download validation and resume behavior."""

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_tcga_brca.py"
SPEC = importlib.util.spec_from_file_location("download_tcga_brca", SCRIPT)
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


class DownloadTCGABRCATest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.raw_dir = root / "raw"
        self.metadata_dir = root / "metadata"
        self.raw_dir.mkdir()
        self.metadata_dir.mkdir()
        self.ledger = self.metadata_dir / "downloaded_files.json"
        self.payload = b"gene_id\ttpm\nENSG001\t1.0\n"
        self.hit = self.manifest_entry("file-one", self.payload)
        self.session = Mock(spec=requests.Session)

    def manifest_entry(self, file_id, payload):
        return {
            "file_id": file_id,
            "file_name": f"{file_id}.rna_seq.augmented_star_gene_counts.tsv",
            "file_size": len(payload),
            "md5sum": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        }

    def local_file(self, hit, payload):
        path = self.raw_dir / hit["file_id"] / hit["file_name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def response_with_files(self, entries):
        members = []
        for hit, payload in entries:
            member = tarfile.TarInfo(f'{hit["file_id"]}/{hit["file_name"]}')
            member.size = len(payload)
            members.append((member, payload))
        return self.response_with_members(members)

    def response_with_members(self, members):
        archive_bytes = io.BytesIO()
        with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
            for member, payload in members:
                archive.addfile(member, io.BytesIO(payload) if member.isfile() else None)
        response = Mock()
        response.iter_content.return_value = [archive_bytes.getvalue()]
        self.session.post.return_value = response
        return response

    def download(self, files=None):
        downloader.download_files(
            self.session,
            files or [self.hit],
            self.raw_dir,
            self.metadata_dir,
            chunk_size=25,
            pause_seconds=0,
        )

    def completed_ids(self):
        return json.loads(self.ledger.read_text())

    def test_invalid_chunk_size_cli_fails_before_filesystem_or_network_access(self):
        root = Path(self.directory.name) / "new-download-root"
        for size in (-1, 0):
            with self.subTest(size=size):
                argv = [str(SCRIPT), "--root", str(root), "--chunk-size", str(size)]
                error = io.StringIO()
                with patch.object(sys, "argv", argv), patch.object(downloader.requests, "Session") as session:
                    with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
                        downloader.main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("--chunk-size must be a positive integer", error.getvalue())
                session.assert_not_called()
                self.assertFalse(root.exists())

    def test_invalid_download_chunk_size_preserves_ledger_before_any_side_effect(self):
        sentinel = b'["preserve-existing-ledger"]\n'
        for size in (-1, 0):
            for files in ([self.hit], []):
                for existing_ledger in (False, True):
                    with self.subTest(size=size, files=files, existing_ledger=existing_ledger):
                        self.ledger.unlink(missing_ok=True)
                        if existing_ledger:
                            self.ledger.write_bytes(sentinel)
                        with patch.object(downloader, "file_matches_manifest") as matches:
                            with self.assertRaisesRegex(ValueError, "chunk_size must be a positive integer"):
                                downloader.download_files(
                                    self.session, files, self.raw_dir, self.metadata_dir, size, pause_seconds=0
                                )
                        matches.assert_not_called()
                        self.session.post.assert_not_called()
                        self.assertEqual(list(self.raw_dir.iterdir()), [])
                        if existing_ledger:
                            self.assertEqual(self.ledger.read_bytes(), sentinel)
                        else:
                            self.assertFalse(self.ledger.exists())

    def test_chunked_requires_positive_size_and_preserves_positive_batching(self):
        for size in (-1, 0):
            for items in ([], ["a", "b"]):
                with self.subTest(size=size, items=items):
                    with self.assertRaisesRegex(ValueError, "chunk size must be a positive integer"):
                        downloader.chunked(items, size)
        self.assertEqual(downloader.chunked(["a", "b", "c"], 1), [["a"], ["b"], ["c"]])
        self.assertEqual(downloader.chunked(["a", "b", "c"], 2), [["a", "b"], ["c"]])
        self.assertEqual(downloader.chunked(["a", "b", "c"], 25), [["a", "b", "c"]])

    def test_regular_files_are_streamed_without_extracting_archive_metadata(self):
        member = tarfile.TarInfo(f'{self.hit["file_id"]}/{self.hit["file_name"]}')
        member.size = len(self.payload)
        member.mode = 0o777
        directory = tarfile.TarInfo(self.hit["file_id"])
        directory.type = tarfile.DIRTYPE
        metadata = tarfile.TarInfo("MANIFEST.txt")
        metadata.size = len(self.payload)
        self.response_with_members([(directory, b""), (metadata, self.payload), (member, self.payload)])

        with patch.object(tarfile.TarFile, "extract", side_effect=AssertionError("Do not extract archive metadata")):
            self.download()

        target = self.raw_dir / self.hit["file_id"] / self.hit["file_name"]
        self.assertEqual(target.read_bytes(), self.payload)
        self.assertEqual(target.stat().st_mode & 0o111, 0)
        self.assertFalse((self.raw_dir / "MANIFEST.txt").exists())
        self.assertEqual(self.completed_ids(), [self.hit["file_id"]])

    def test_unsafe_archive_paths_are_rejected_before_extraction(self):
        names = [
            "../outside.tsv",
            f"{self.raw_dir.parent}/outside.tsv",
            f'{self.hit["file_id"]}/../../outside.tsv',
            f'{self.hit["file_id"]}/nested/{self.hit["file_name"]}',
            "..\\outside.tsv/payload.tsv",
        ]
        for name in names:
            with self.subTest(name=name):
                member = tarfile.TarInfo(name)
                member.size = len(self.payload)
                self.response_with_members([(member, self.payload)])
                # Also prevents the old implementation from writing to an absolute
                # filesystem path when demonstrating this regression.
                with patch.object(tarfile.TarFile, "extract") as unsafe_extract:
                    with self.assertRaisesRegex(ValueError, "archive member"):
                        self.download()
                unsafe_extract.assert_not_called()
                self.assertFalse((self.raw_dir.parent / "outside.tsv").exists())
                self.assertEqual(self.completed_ids(), [])

    def test_archive_payload_must_match_the_requested_manifest(self):
        names = [f'other-id/{self.hit["file_name"]}', f'{self.hit["file_id"]}/other.tsv']
        for name in names:
            with self.subTest(name=name):
                member = tarfile.TarInfo(name)
                member.size = len(self.payload)
                self.response_with_members([(member, self.payload)])
                with self.assertRaisesRegex(ValueError, "manifest"):
                    self.download()
                self.assertFalse((self.raw_dir / name).exists())
                self.assertEqual(self.completed_ids(), [])

    def test_archive_links_are_rejected_without_following_their_targets(self):
        for member_type in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            with self.subTest(member_type=member_type):
                member = tarfile.TarInfo(f'{self.hit["file_id"]}/{self.hit["file_name"]}')
                member.type = member_type
                member.linkname = "../../outside.tsv"
                self.response_with_members([(member, b"")])
                with self.assertRaisesRegex(ValueError, "archive member"):
                    self.download()
                self.assertFalse((self.raw_dir.parent / "outside.tsv").exists())
                self.assertEqual(self.completed_ids(), [])

    def test_existing_symlinks_cannot_redirect_downloads_outside_raw_directory(self):
        outside = self.raw_dir.parent / "outside"
        outside.mkdir()
        sentinel = outside / self.hit["file_name"]
        sentinel.write_bytes(b"keep me")
        file_dir = self.raw_dir / self.hit["file_id"]
        for link_directory in (True, False):
            with self.subTest(link_directory=link_directory):
                if link_directory:
                    link = file_dir
                    link.symlink_to(outside, target_is_directory=True)
                else:
                    file_dir.mkdir()
                    link = file_dir / self.hit["file_name"]
                    link.symlink_to(sentinel)
                try:
                    self.response_with_files([(self.hit, self.payload)])
                    with self.assertRaisesRegex(ValueError, "download directory"):
                        self.download()
                    self.assertEqual(sentinel.read_bytes(), b"keep me")
                    self.session.post.assert_not_called()
                finally:
                    link.unlink()
                    if not link_directory:
                        file_dir.rmdir()
                    self.session.reset_mock()

    def test_manifest_path_components_are_validated_before_network_access(self):
        cases = [
            ("file_id", ".."),
            ("file_id", "../outside"),
            ("file_id", str(self.raw_dir.parent / "outside")),
            ("file_name", "../outside.tsv"),
            ("file_name", str(self.raw_dir.parent / "outside.tsv")),
            ("file_name", "..\\outside.tsv"),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                hit = dict(self.hit, **{field: value})
                self.response_with_files([])
                with self.assertRaisesRegex(ValueError, "manifest"):
                    self.download([hit])
                self.session.post.assert_not_called()
                self.session.reset_mock()

    def test_stale_ledger_does_not_skip_missing_file(self):
        self.ledger.write_text(json.dumps([self.hit["file_id"]]))
        self.response_with_files([(self.hit, self.payload)])

        self.download()

        self.session.post.assert_called_once_with(
            f"{downloader.GDC_API}/data",
            json={"ids": [self.hit["file_id"]]},
            timeout=600,
            stream=True,
        )
        self.assertTrue(downloader.file_matches_manifest(self.hit, self.raw_dir))
        self.assertEqual(self.completed_ids(), [self.hit["file_id"]])

    def test_truncated_local_file_is_downloaded_again(self):
        self.ledger.write_text(json.dumps([self.hit["file_id"]]))
        path = self.local_file(self.hit, self.payload[:-1])
        self.response_with_files([(self.hit, self.payload)])

        self.download()

        self.session.post.assert_called_once()
        self.assertEqual(path.read_bytes(), self.payload)

    def test_same_size_local_file_with_wrong_md5_is_downloaded_again(self):
        self.ledger.write_text(json.dumps([self.hit["file_id"]]))
        path = self.local_file(self.hit, b"x" * len(self.payload))
        self.response_with_files([(self.hit, self.payload)])

        self.download()

        self.session.post.assert_called_once()
        self.assertEqual(path.read_bytes(), self.payload)

    def test_valid_local_file_is_preserved_with_or_without_ledger(self):
        self.local_file(self.hit, self.payload)
        for ledger_entries in (None, [self.hit["file_id"], "obsolete-file"]):
            with self.subTest(ledger_entries=ledger_entries):
                self.ledger.unlink(missing_ok=True)
                if ledger_entries is not None:
                    self.ledger.write_text(json.dumps(ledger_entries))

                self.download()

                self.session.post.assert_not_called()
                self.assertEqual(self.completed_ids(), [self.hit["file_id"]])

    def test_size_validation_still_applies_without_md5(self):
        hit = dict(self.hit)
        del hit["md5sum"]
        self.local_file(hit, self.payload)

        self.download([hit])

        self.session.post.assert_not_called()
        self.assertEqual(self.completed_ids(), [hit["file_id"]])

    def test_invalid_download_is_not_recorded_as_complete(self):
        for returned_payload in (self.payload[:-1], b"x" * len(self.payload), None):
            with self.subTest(returned_payload=returned_payload):
                path = self.raw_dir / self.hit["file_id"] / self.hit["file_name"]
                path.unlink(missing_ok=True)
                self.ledger.write_text(json.dumps([self.hit["file_id"]]))
                entries = [] if returned_payload is None else [(self.hit, returned_payload)]
                self.response_with_files(entries)

                with self.assertRaisesRegex(RuntimeError, "failed manifest validation: file-one"):
                    self.download()

                self.assertEqual(self.completed_ids(), [])

    def test_partial_chunk_records_only_verified_files(self):
        other = self.manifest_entry("file-two", self.payload)
        self.response_with_files([(self.hit, self.payload), (other, self.payload[:-1])])

        with self.assertRaisesRegex(RuntimeError, "failed manifest validation: file-two"):
            self.download([self.hit, other])

        self.assertEqual(self.completed_ids(), [self.hit["file_id"]])

    def test_http_failure_clears_stale_completion_entry(self):
        self.ledger.write_text(json.dumps([self.hit["file_id"]]))
        response = self.response_with_files([])
        response.raise_for_status.side_effect = requests.HTTPError("download failed")

        with self.assertRaises(requests.HTTPError):
            self.download()

        self.assertEqual(self.completed_ids(), [])


if __name__ == "__main__":
    unittest.main()
