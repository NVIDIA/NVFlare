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

"""PyTorch Lightning API integration module for simplified imports.

Usage:
    from nvflare.client.lightning import patch

For detailed information on usage and the API, refer to:
    :mod:`nvflare.app_opt.lightning.api`

"""

from nvflare.fuel.utils.import_utils import optional_import

pytorch_lightning, ok = optional_import(module="pytorch_lightning")

if ok:
    from nvflare.apis.analytix import AnalyticsDataType as AnalyticsDataType
    from nvflare.app_common.abstract.fl_model import FLModel as FLModel
    from nvflare.app_common.abstract.fl_model import ParamsType as ParamsType
    from nvflare.app_opt.lightning import FLCallback as FLCallback
    from nvflare.app_opt.lightning import patch as patch
    from nvflare.client import get_config as get_config
    from nvflare.client import get_job_id as get_job_id
    from nvflare.client import get_site_name as get_site_name
    from nvflare.client import is_running as is_running
    from nvflare.client import log as log
    from nvflare.client import receive as receive
    from nvflare.client import system_info as system_info
