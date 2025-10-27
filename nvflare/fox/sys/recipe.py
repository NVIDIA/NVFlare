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
from nvflare.fox.api.app import App, ClientApp, ServerApp
from nvflare.fox.api.filter import FilterChain
from nvflare.fox.api.strategy import Strategy
from nvflare.fuel.utils.validation_utils import check_object_type, check_positive_int, check_positive_number, check_str
from nvflare.job_config.api import FedJob
from nvflare.recipe.spec import Recipe

from .controller import FoxController
from .executor import FoxExecutor


class FoxRecipe(Recipe):

    def __init__(
        self,
        job_name: str,
        server_app: ServerApp,
        client_app: ClientApp,
        sync_task_timeout=5,
        max_call_threads_for_server=100,
        max_call_threads_for_client=100,
        min_clients: int = 1,
    ):
        check_str("job_name", job_name)
        check_object_type("server_app", server_app, ServerApp)
        check_positive_number("sync_task_timeout", sync_task_timeout)
        check_positive_int("max_call_threads_for_server", max_call_threads_for_server)
        check_positive_int("max_call_threads_for_client", max_call_threads_for_client)
        check_positive_int("min_clients", min_clients)

        if not isinstance(client_app, ClientApp):
            raise ValueError(f"client_app must be ClientApp but got {type(client_app)}")

        # make sure server app has strategy
        if not server_app.strategies:
            raise ValueError(f"server_app has no strategies")

        self.job_name = job_name
        self.server_app = server_app
        self.client_app = client_app
        self.sync_task_timeout = sync_task_timeout
        self.max_call_threads_for_server = max_call_threads_for_server
        self.max_call_threads_for_client = max_call_threads_for_client
        self.min_clients = min_clients

        job = self._create_job()
        Recipe.__init__(self, job)

    def _create_job(self) -> FedJob:
        job = FedJob(name=self.job_name, min_clients=self.min_clients)

        server_app_id = job.to_server(self.server_app, "_app")

        # get all strategies
        strategy_ids = []
        for name, strategy in self.server_app.strategies:
            comp_id = job.to_server(strategy, id=name)
            strategy_ids.append(comp_id)

        collab_obj_ids, in_cf_arg, out_cf_arg, in_rf_arg, out_rf_arg = self._create_app_args(
            self.server_app, job.to_server
        )

        controller = FoxController(
            strategy_ids=strategy_ids,
            server_app_id=server_app_id,
            collab_obj_ids=collab_obj_ids,
            incoming_call_filters=in_cf_arg,
            outgoing_call_filters=out_cf_arg,
            incoming_result_filters=in_rf_arg,
            outgoing_result_filters=out_rf_arg,
            sync_task_timeout=self.sync_task_timeout,
            max_call_threads=self.max_call_threads_for_server,
            props=self.server_app.get_props(),
            resource_dirs=self.server_app.get_resource_dirs(),
        )

        job.to_server(controller, id="controller")

        # add client config
        client_app_id = job.to_clients(self.client_app, "_app")
        c_collab_obj_ids, c_in_cf_arg, c_out_cf_arg, c_in_rf_arg, c_out_rf_arg = self._create_app_args(
            self.client_app, job.to_clients
        )
        executor = FoxExecutor(
            client_app_id=client_app_id,
            collab_obj_ids=c_collab_obj_ids,
            incoming_call_filters=c_in_cf_arg,
            outgoing_call_filters=c_out_cf_arg,
            incoming_result_filters=c_in_rf_arg,
            outgoing_result_filters=c_out_rf_arg,
            max_call_threads=self.max_call_threads_for_client,
            props=self.client_app.get_props(),
            resource_dirs=self.client_app.get_resource_dirs(),
        )
        job.to_clients(executor, id="executor", tasks=["*"])
        return job

    def _create_app_args(self, app: App, to_f):
        # collab objs
        collab_obj_ids = []
        collab_objs = app.get_collab_objects()
        for name, obj in collab_objs.items():
            if isinstance(obj, Strategy):
                # do not include strategy in collab objs since it's done separately.
                continue
            comp_id = to_f(obj, id=name)
            collab_obj_ids.append(comp_id)

        # build filter components
        # since a filter object could be used multiple times, we must make sure that only one component is created
        # for the same object!
        filter_comp_table = {}
        incoming_call_filters = app.get_incoming_call_filters()
        outgoing_call_filters = app.get_outgoing_call_filters()
        incoming_result_filters = app.get_incoming_result_filters()
        outgoing_result_filters = app.get_outgoing_result_filters()

        self._create_filter_components(to_f, incoming_call_filters, filter_comp_table)
        self._create_filter_components(to_f, outgoing_call_filters, filter_comp_table)
        self._create_filter_components(to_f, incoming_result_filters, filter_comp_table)
        self._create_filter_components(to_f, outgoing_result_filters, filter_comp_table)

        # filters
        in_cf_arg = self._create_filer_chain_arg(incoming_call_filters, filter_comp_table)
        out_cf_arg = self._create_filer_chain_arg(outgoing_call_filters, filter_comp_table)
        in_rf_arg = self._create_filer_chain_arg(incoming_result_filters, filter_comp_table)
        out_rf_arg = self._create_filer_chain_arg(outgoing_result_filters, filter_comp_table)
        return collab_obj_ids, in_cf_arg, out_cf_arg, in_rf_arg, out_rf_arg

    @staticmethod
    def _create_filer_chain_arg(filter_chains: list, comp_table: dict):
        result = []
        for chain in filter_chains:
            assert isinstance(chain, FilterChain)
            filter_ids = []
            for f in chain.filters:
                comp_id = comp_table[id(f)]
                filter_ids.append(comp_id)
            d = {"pattern": chain.pattern, "filters": filter_ids}
            result.append(d)
        return result

    @staticmethod
    def _create_filter_components(to_f, filter_chains: list, comp_table: dict):
        for chain in filter_chains:
            assert isinstance(chain, FilterChain)
            for f in chain.filters:
                fid = id(f)
                comp_id = comp_table.get(fid)
                if not comp_id:
                    comp_id = to_f(f, id="_filter")
                    comp_table[fid] = comp_id
