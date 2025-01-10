# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import os
from typing import Any

from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs import Decomposer
from nvflare.fuel.utils.fobs.datum import DatumManager
from nvflare.fuel.utils.fobs.fobs import register_custom_folder
from nvflare.private.fed.utils.fed_utils import extract_participants
from nvflare.utils.job_launcher_utils import extract_job_image


class ExampleTestClass:
    def __init__(self, number):
        self.number = number


class ExampleTestClassDecomposer(Decomposer):
    def __init__(self):
        super().__init__()
        self.type = ExampleTestClass.__name__

    def __eq__(self, other):
        return hasattr(other, "type") and self.type == other.type

    def supported_type(self):
        return ExampleTestClass

    def decompose(self, target: ExampleTestClass, manager: DatumManager = None) -> Any:
        return target.number

    def recompose(self, data: Any, manager: DatumManager = None) -> ExampleTestClass:
        return ExampleTestClass(data)


class TestFedUtils:
    def test_custom_fobs_initialize(self):
        pwd = os.path.dirname(os.path.realpath(__file__))
        register_custom_folder(pwd)
        decomposer = ExampleTestClassDecomposer()
        decomposers = fobs.fobs._decomposers
        assert decomposer in list(decomposers.values())

    def test_extract_participants(self):
        participants = ["site-1", "site-2"]
        results = extract_participants(participants)
        expected = ["site-1", "site-2"]
        assert results == expected

        participants = ["@ALL"]
        results = extract_participants(participants)
        expected = ["@ALL"]
        assert results == expected

    def test_extract_participants_with_image(self):
        participants = [
            "site-1",
            "site-2",
            {"sites": ["site-3", "site-4"], "image": "image1"},
            {"sites": ["site-5"], "image": "image2"},
        ]
        results = extract_participants(participants)
        expected = ["site-1", "site-2", "site-3", "site-4", "site-5"]
        assert results == expected

    def test_extract_job_image(self):
        job_meta = {"deploy_map": {"app": ["site-1", "site-2", {"sites": ["site-3", "site-4"], "image": "image1"}]}}
        result = extract_job_image(job_meta, "site-3")
        expected = "image1"
        assert result == expected

        result = extract_job_image(job_meta, "site-1")
        expected = None
        assert result == expected

        job_meta = {"deploy_map": {"app": ["site-1", "site-2"]}}
        result = extract_job_image(job_meta, "site-1")
        expected = None
        assert result == expected
