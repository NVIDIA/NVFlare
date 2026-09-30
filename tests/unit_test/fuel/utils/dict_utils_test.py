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
import unittest

from nvflare.fuel.utils.dict_utils import augment


class TestAugment(unittest.TestCase):
    def test_nested_override_replaces_the_value(self):
        to = {"a": {"x": 1}}
        err = augment(to, {"a": {"x": 2}}, from_override_to=True)
        self.assertEqual(err, "")
        self.assertEqual(to["a"]["x"], 2)

    def test_nested_value_stays_when_override_is_off(self):
        to = {"a": {"x": 1}}
        err = augment(to, {"a": {"x": 2}}, from_override_to=False)
        self.assertEqual(err, "")
        self.assertEqual(to["a"]["x"], 1)

    def test_nested_append_list_is_honored(self):
        to = {"n": {"widgets": [{"id": 1}]}}
        err = augment(to, {"n": {"widgets": [{"id": 2}]}}, append_list="widgets")
        self.assertEqual(err, "")
        self.assertEqual(to["n"]["widgets"], [{"id": 1}, {"id": 2}])

    def test_nested_components_still_append(self):
        to = {"n": {"components": [{"id": 1}]}}
        err = augment(to, {"n": {"components": [{"id": 2}]}})
        self.assertEqual(err, "")
        self.assertEqual(to["n"]["components"], [{"id": 1}, {"id": 2}])


if __name__ == "__main__":
    unittest.main()
