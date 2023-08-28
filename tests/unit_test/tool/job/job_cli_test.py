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
from unittest.mock import patch

import pytest
from pyhocon import ConfigFactory as CF

from nvflare.fuel_opt.utils.pyhocon_loader import PyhoconConfig
from nvflare.tool.job.job_cli import _update_client_app_config_script, convert_args_list_to_dict


class TestJobCLI:
    @pytest.mark.parametrize("inputs, result", [(["a=1", "b=2", "c = 3"], dict(a="1", b="2", c="3"))])
    def test_convert_args_list_to_dict(self, inputs, result):
        r = convert_args_list_to_dict(inputs)
        assert r == result

    def test__update_client_app_config_script(self):
        with patch("nvflare.fuel.utils.config_factory.ConfigFactory.load_config") as mock2:
            conf = CF.parse_string(
                """ 
                    {
                      format_version = 2
                      app_script = "python custom/cifar10.py"
                      app_config = ""
                      executors = [
                        {
                          tasks = ["train"]
                          executor {
                            name = "PTFilePipeLauncherExecutor"
                            args {
                              launcher_id = "launcher"
                              heartbeat_timeout = 60
                            }
                          }
                        }
                      ],
                      task_result_filters= []
                      task_data_filters =  []
                      components =  [
                        {
                          id = "launcher"
                          name =  "SubprocessLauncher"
                          args.script = "{app_script}  {app_config} "
                        }
                      ]
                    }
                """
            )
            mock2.return_value = PyhoconConfig(conf, "/tmp/my_job/app/config/config_fed_client.conf")

            app_config = ["trainer.batch_size=1024", "eval_iters=100", "lr=0.1"]
            config, config_path = _update_client_app_config_script("/tmp/my_job/", app_config)
            assert config.get("app_config") == "--trainer.batch_size=1024 --eval_iters=100 --lr=0.1"
