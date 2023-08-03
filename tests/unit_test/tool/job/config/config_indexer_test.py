# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nvflare.tool.job.config.config_indexer import build_dict_reverse_order_index, expand_indices


class TestConfigIndex:
    def test_dict_indexer(self):
        key_paths = build_dict_reverse_order_index(config_dict={})
        assert len(key_paths) == 0

        config_dict = dict(
            x=dict(
                x1=dict(x11=2),
                x2=dict(x21=3, x22=4, x23=dict(x31=3, x32=4)),
                y=dict(y1=dict(y11=2), y2=dict(y21=1)),
                z=[
                    dict(id=2, x1=dict(x11=2), x2=dict(x21=3, x22=4)),
                    dict(id=3, y1=dict(y11=2), y2=dict(y21=1)),
                    dict(id=4, z1=dict(z11=2), z2=dict(z21=1)),
                    100,
                ],
            )
        )

        expected_key_paths = {
            "x31": ["x.x2.x23.x31"],
            "x32": ["x.x2.x23.x32"],
            "x22": ["x.x2.x22", "x.z[0].x2.x22"],
            "x21": ["x.x2.x21", "x.z[0].x2.x21"],
            "x11": ["x.x1.x11", "x.z[0].x1.x11"],
            "y21": ["x.y.y2.y21", "x.z[1].y2.y21"],
            "y11": ["x.y.y1.y11", "x.z[1].y1.y11"],
            "z21": ["x.z[2].z2.z21"],
            "z11": ["x.z[2].z1.z11"],
            "z[3]": ["x.z[3]"],
            "id": ["x.z[0].id", "x.z[1].id", "x.z[2].id"],
        }

        key_paths = build_dict_reverse_order_index(config_dict=config_dict)
        for key in key_paths:
            print(key, key_paths[key])

        diff1 = set(key_paths.keys()) - set(expected_key_paths.keys())
        diff2 = set(expected_key_paths.keys()) - set(key_paths.keys())

        assert len(diff2) == 0
        assert len(diff1) == 0

        for key in expected_key_paths:
            assert key_paths[key] == expected_key_paths[key]

        indices = expand_indices(key_paths)
        for key in indices:
            print("key=", key, ":", indices[key])

        assert indices["z[2].z1.z11"] == ["x.z[2].z1.z11"]
        assert len(indices) == 52
