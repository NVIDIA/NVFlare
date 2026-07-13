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


class CollabAdaptor:

    def __init__(
        self,
        collab_obj_ids: List[str] = None,
        props: Dict[str, Any] = None,
        resource_dirs: Dict[str, str] = None,
        incoming_call_filters=None,
        outgoing_call_filters=None,
        incoming_result_filters=None,
        outgoing_result_filters=None,
    ):
        if not collab_obj_ids:
            collab_obj_ids = []
        self.props = props
        self.resource_dirs = resource_dirs
        self.publish_obj_ids = collab_obj_ids
        self.incoming_call_filters = incoming_call_filters
        self.outgoing_call_filters = outgoing_call_filters
        self.incoming_result_filters = incoming_result_filters
        self.outgoing_result_filters = outgoing_result_filters

    def process_config(self, app: App, fl_ctx: FLContext):
        app.update_props(self.props)
        app.set_resource_dirs(self.resource_dirs)

        engine = fl_ctx.get_engine()
        if self.publish_obj_ids:
            for cid in self.publish_obj_ids:
                obj = engine.get_component(cid)
                if not obj:
                    return f"component {cid} does not exist"

                app.add_collab_object(cid, obj)

        err = self._parse_filters("incoming_call_filters", app.add_incoming_call_filters, fl_ctx)
        if err:
            return err

        err = self._parse_filters("outgoing_call_filters", app.add_outgoing_call_filters, fl_ctx)
        if err:
            return err

        err = self._parse_filters("incoming_result_filters", app.add_incoming_result_filters, fl_ctx)
        if err:
            return err

        err = self._parse_filters("outgoing_result_filters", app.add_outgoing_result_filters, fl_ctx)
        if err:
            return err

        return None

    def _parse_filters(self, name, add_f, fl_ctx):
        filters = getattr(self, name)
        if not filters:
            return None

        if not isinstance(filters, list):
            return f"{name} must be a list but got {type(filters)}"

        for chain_dict in filters:
            pattern, filter_components, err = self._parse_filter_chain(name, chain_dict, fl_ctx)
            if err:
                return err
            add_f(pattern, filter_components)
        return None

    @staticmethod
    def _parse_filter_chain(chain_name, chain_dict: dict, fl_ctx):
        if not isinstance(chain_dict, dict):
            return None, None, f"element in {chain_name} must be dict but got {type(chain_dict)}"

        pattern = chain_dict.get("pattern")
        if not pattern:
            return None, None, f"missing 'pattern' in {chain_name}"

        filter_ids = chain_dict.get("filters")
        if not filter_ids:
            return None, None, f"missing 'filters' in {chain_name}"

        if not isinstance(filter_ids, list):
            return None, None, f"invalid 'filters' in {chain_name}: expect list got {type(filter_ids)}"

        engine = fl_ctx.get_engine()
        filters = []
        for fid in filter_ids:
            f = engine.get_component(fid)
            if not f:
                return None, None, f"component {fid} does not exist"
            filters.append(f)
        return pattern, filters, None
