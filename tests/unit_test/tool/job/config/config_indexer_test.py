# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from pyhocon import ConfigFactory as CF

from nvflare.tool.job.config.config_indexer import KeyIndex, build_dict_reverse_order_index
from nvflare.tool.job.config.configer import extract_string_with_index, filter_config_name_and_values


class TestConfigIndex:
    def test_dict_indexer(self):
        key_indices = build_dict_reverse_order_index(config=CF.from_dict({}))
        assert len(key_indices) == 0

        config_dict = dict(
            x=dict(
                x1=dict(x11=2),
                x2=dict(x21=3, x22=4, x23=dict(x31=3, x32=4)),
                y=dict(y1=dict(y11=2), y2=dict(y21=1)),
                z=[
                    dict(id=2),
                    dict(id=3),
                    dict(id=4),
                    100,
                ],
                s=[
                    dict(id=2),
                    100,
                ],
            )
        )

        config = CF.from_dict(config_dict)
        x_key = KeyIndex(key="x", value=config.get("x"), parent_key=None)

        x1_key = KeyIndex(key="x1", value=config.get("x").get("x1"), parent_key=x_key)
        x2_key = KeyIndex(key="x2", value=config.get("x").get("x2"), parent_key=x_key)
        y_key = KeyIndex(key="y", value=config.get("x").get("y"), parent_key=x_key)
        z_key = KeyIndex(key="z", value=config.get("x").get("z"), parent_key=x_key)
        s_key = KeyIndex(key="s", value=config.get("x").get("s"), parent_key=x_key)

        x11_key = KeyIndex(key="x11", value=2, parent_key=x1_key)

        x21_key = KeyIndex(key="x21", value=3, parent_key=x2_key)
        x22_key = KeyIndex(key="x22", value=4, parent_key=x2_key)
        x23_key = KeyIndex(key="x23", value=config.get("x").get("x2").get("x23"), parent_key=x2_key)

        x31_key = KeyIndex(key="x31", value=3, parent_key=x23_key)
        x32_key = KeyIndex(key="x32", value=4, parent_key=x23_key)

        y1_key = KeyIndex(key="y1", value=config.get("x").get("y").get("y1"), parent_key=y_key)
        y2_key = KeyIndex(key="y2", value=config.get("x").get("y").get("y2"), parent_key=y_key)

        y11_key = KeyIndex(key="y11", value=2, parent_key=y1_key)
        y21_key = KeyIndex(key="y21", value=1, parent_key=y2_key)

        z0_key = KeyIndex(key="z[0]", value=config.get("x").get("z")[0], parent_key=z_key, index=0)
        z1_key = KeyIndex(key="z[1]", value=config.get("x").get("z")[1], parent_key=z_key, index=1)
        z2_key = KeyIndex(key="z[2]", value=config.get("x").get("z")[2], parent_key=z_key, index=2)
        z3_key = KeyIndex(key="z[3]", value=100, parent_key=z_key, index=3)

        s0_key = KeyIndex(key="s[0]", value=config.get("x").get("s")[0], parent_key=s_key, index=0)
        s1_key = KeyIndex(key="s[1]", value=100, parent_key=s_key, index=1)

        id_keys = [
            KeyIndex(key="id", value=2, parent_key=z0_key),
            KeyIndex(key="id", value=3, parent_key=z1_key),
            KeyIndex(key="id", value=4, parent_key=z2_key),
            KeyIndex(key="id", value=2, parent_key=s0_key),
        ]

        expected_keys = {
            "x31": x31_key,
            "x32": x32_key,
            "x22": x22_key,
            "x21": x21_key,
            "x11": x11_key,
            "y21": y21_key,
            "y11": y11_key,
            "z[3]": z3_key,
            "id": id_keys,
            "s[1]": s1_key,
        }

        key_indices = build_dict_reverse_order_index(config=CF.from_dict(config_dict))

        print("\n\n")
        for key in key_indices:
            e = expected_keys[key]
            b_list = key_indices[key]
            if len(b_list) == 1:
                b = b_list[0]
                a = e
                assert key == a.key
                assert key == b.key
                assert a.key == b.key and a.value == b.value
                assert a.index == b.index
                if b.component_name is None or b.component_name.strip() == "":
                    assert a.component_name is None or a.component_name.strip() == ""
                else:
                    assert a.component_name == b.component_name

                assert a.parent_key.key == b.parent_key.key
                assert a.parent_key.value == b.parent_key.value
                assert a.parent_key.index == b.parent_key.index
            else:
                xs = zip(e, b_list)
                for a, b in xs:
                    assert a.key == b.key and a.value == b.value
                    assert a.index == b.index
                    if b.component_name is None or b.component_name.strip() == "":
                        assert a.component_name is None or a.component_name.strip() == ""
                    else:
                        assert a.component_name == b.component_name

                    assert a.parent_key.key == b.parent_key.key
                    assert a.parent_key.value == b.parent_key.value
                    assert a.parent_key.index == b.parent_key.index

        diff1 = set(key_indices.keys()) - set(expected_keys.keys())
        diff2 = set(expected_keys.keys()) - set(key_indices.keys())

        assert len(diff2) == 0
        assert len(diff1) == 0

    def test_extract_string_with_index(self):
        input_string = "components[0].args.data_path"
        tokens = extract_string_with_index(input_string)
        assert tokens == [("components", 0, ["args.data_path"])]

    def test_extract_file_from_dict_by_index(self):
        config_str = """
                    {
                        "components": [
                            {
                              "id": "df_stats_generator",
                              "path": "df_statistics.DFStatistics",
                              "args": {
                                "data_path": "data.csv"
                                "other_path": null
                              }
                            },
                            {
                              "id": "min_max_cleanser",
                              "path": "nvflare.app_common.statistics.min_max_cleanser.AddNoiseToMinMax",
                              "args": {
                                "min_noise_level": 0.1,
                                "max_noise_level": 0.3
                              }
                            }
                        ]
                    }
                    """
        conf = CF.parse_string(config_str)
        key_indices = build_dict_reverse_order_index(config=conf)
        exclude_key_list = []
        result = filter_config_name_and_values(exclude_key_list, key_indices)
        key_index = result["data_path"]
        assert key_index.value == "data.csv"
        key_index = result["other_path"]
        assert key_index.value is None
