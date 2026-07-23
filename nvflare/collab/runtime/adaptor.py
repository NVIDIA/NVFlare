# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any, Dict, List

from nvflare.apis.fl_context import FLContext
from nvflare.collab.api.app import App
from nvflare.collab.api.constants import FL_CONTEXT_PROP


class CollabAdaptor:

    def __init__(
        self,
        collab_obj_ids: List[str] = None,
        props: Dict[str, Any] = None,
    ):
        if not collab_obj_ids:
            collab_obj_ids = []
        self.props = props
        self.publish_obj_ids = collab_obj_ids

    def process_config(self, app: App, fl_ctx: FLContext):
        app.update_props(self.props)
        # FLContext is a live runtime object and cannot be serialized into the
        # recipe config. Add it after the site app is created so every Collab
        # function at this site can access it through the facade.
        app.set_prop(FL_CONTEXT_PROP, fl_ctx)
        engine = fl_ctx.get_engine()
        if self.publish_obj_ids:
            for cid in self.publish_obj_ids:
                obj = engine.get_component(cid)
                if not obj:
                    return f"component {cid} does not exist"

                app.add_collab_object(cid, obj)

        return None
