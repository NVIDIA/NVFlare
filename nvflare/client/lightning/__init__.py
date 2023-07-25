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

import warnings

from nvflare.fuel.utils.import_utils import optional_import

pytorch_lightning, ok = optional_import(module="pytorch_lightning")

if ok:
    from nvflare.app_opt.lightning import init as init
    from nvflare.app_opt.lightning import patch as patch
    from nvflare.client import get_sys_meta as get_sys_meta
    from nvflare.client import send_model as send_model
    from nvflare.client import submit_metrics as submit_metrics
else:
    warnings.warn("pytorch_lightning is not installed, nvflare.client.lightning will not work.")
