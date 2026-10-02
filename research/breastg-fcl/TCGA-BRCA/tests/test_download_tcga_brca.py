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

import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

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
        archive_bytes = io.BytesIO()
        with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
            for hit, payload in entries:
                member = tarfile.TarInfo(f'{hit["file_id"]}/{hit["file_name"]}')
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
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
