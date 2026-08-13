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

"""Force CSE to request the attached trainer's live local model."""

from nvflare.app_common.ccwf.comps.np_file_model_persistor import NPFileModelPersistor


class EmptyModelPersistor(NPFileModelPersistor):
    def get_model_inventory(self, fl_ctx):
        return {}

    def handle_event(self, event, fl_ctx):
        pass
