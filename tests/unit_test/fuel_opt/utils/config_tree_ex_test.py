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

from nvflare.fuel_opt.utils.config_tree_ex import ConfigTreeEx


class TestConfigTreeEx:
    def test_to_hocon_with_comments(self):
        conf_str = """
        {
            workflows {
                SAG.workflow { 
                    id = "scatter_and_gather"
                    name = "ScatterAndGather"
                }
                SAG.components{
                     persistor {
                          id =  "persistor"
                          name =  "PTFileModelPersistor"
                    }
                }
                SAG.executors { 
                    PTFilePipeLauncherExecutor {
                        executor {
                            tasks = ["train"]
                            executor {
                                name = "PTFilePipeLauncherExecutor"
                            }
                        }
                    }
                }
            }
        }
        """

        config = CF.parse_string(conf_str)
        workflows_conf = config.get("workflows")
        sag_config = workflows_conf.get("SAG")
        workflows = []
        for name, item in sag_config.items():
            item = ConfigTreeEx(item)
            item.put_comment(f" this is {name}")
            workflows.append(item)

        output_config = CF.parse_string("""{ format_version = 2 }""")
        output_config.put("workflows", workflows)

        s = ConfigTreeEx(output_config).to_hocon()
        for name, _ in sag_config.items():
            comment = f" this is {name}"
            assert comment in s
