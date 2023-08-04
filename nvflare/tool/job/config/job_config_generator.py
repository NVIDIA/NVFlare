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


def create_job(cmd_args):
    prepare_job_folder(cmd_args)
    predefined = load_predefined_config()
    prepare_fed_config(cmd_args, predefined)
    prepare_meta_config(cmd_args)
    prepare_model_exchange_config(cmd_args, predefined)
