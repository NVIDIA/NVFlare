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

import json
from pathlib import Path

import pytest

from nvflare.app_common.default_component_policy import DEFAULT_CLASS_ALLOW_LIST
from nvflare.lighter.constants import CtxKey
from nvflare.lighter.entity import Participant, Project
from nvflare.lighter.impl.static_file import StaticFileBuilder


class _FakeCtx:
    def __init__(self, project, root_dir):
        self._project = project
        self._root_dir = root_dir
        self.calls = []

    def get_project(self):
        return self._project

    def get(self, key):
        return {
            CtxKey.ADMIN_PORT: 8002,
            CtxKey.FED_LEARN_PORT: 8003,
        }.get(key)

    def get_kit_dir(self, entity):
        return self._root_dir / entity.name / "startup"

    def get_local_dir(self, entity):
        return self._root_dir / entity.name / "local"

    def get_ws_dir(self, entity):
        return self._root_dir / entity.name

    def build_from_template(
        self, dest_dir, temp_section, file_name, replacement=None, mode="t", exe=False, content_modify_cb=None, **kwargs
    ):
        self.calls.append((str(dest_dir), file_name))


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _extract_class_allow_list(resource_template):
    """Extract the class_allow_list JSON array verbatim.

    Parses the embedded JSON array so trailing-dot package prefixes (which
    the previous regex-based extractor silently dropped) are included.
    """
    list_pos = resource_template.index('"class_allow_list"')
    start = resource_template.index("[", list_pos)
    depth = 0
    end = None
    for i in range(start, len(resource_template)):
        if resource_template[i] == "[":
            depth += 1
        elif resource_template[i] == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    return json.loads(resource_template[start : end + 1])


class TestStaticFileBuilder:
    @pytest.mark.parametrize(
        "scheme",
        [("grpc"), ("http"), ("tcp")],
    )
    def test_scheme(self, scheme):
        builder = StaticFileBuilder(scheme=scheme)
        assert builder.scheme == scheme

    def test_build_server_emits_study_registry_when_studies_exist(self, tmp_path):
        server = Participant(type="server", name="server1", org="org")
        project = Project(
            name="proj",
            description="desc",
            participants=[server],
            props={
                "api_version": 4,
                "studies": {
                    "study_a": {
                        "site_orgs": {"org": ["client1"]},
                        "admins": ["admin1@org.com"],
                    }
                },
            },
        )
        ctx = _FakeCtx(project=project, root_dir=tmp_path)

        builder = StaticFileBuilder()
        builder._build_server(project.get_server(), ctx)

        registry_path = tmp_path / server.name / "local" / "study_registry.json"
        assert registry_path.exists()
        assert json.loads(registry_path.read_text()) == {
            "format_version": "1.0",
            "studies": project.get_prop("studies"),
        }

    def test_build_server_does_not_emit_study_registry_without_studies(self, tmp_path):
        server = Participant(type="server", name="server1", org="org")
        project = Project(
            name="proj",
            description="desc",
            participants=[server],
            props={
                "api_version": 4,
            },
        )
        ctx = _FakeCtx(project=project, root_dir=tmp_path)

        builder = StaticFileBuilder()
        builder._build_server(project.get_server(), ctx)

        registry_path = tmp_path / server.name / "local" / "study_registry.json"
        assert not registry_path.exists()

    def test_master_template_moves_user_config_runtime_workspace(self):
        """CC startup kits live in plaintext /user_config, so runtime artifacts must not default there."""
        import os

        import yaml

        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "nvflare",
            "lighter",
            "templates",
            "master_template.yml",
        )

        with open(template_path, "r") as f:
            template = yaml.safe_load(f)

        sub_start = template["sub_start_sh"]
        stop_fl = template["stop_fl_sh"]
        copy_cmd = 'cp -a "$SOURCE_WORKSPACE" "$KIT_WORKSPACE"'
        verify_cmd = 'verify_startup_kits -f "$KIT_WORKSPACE" -c "$KIT_WORKSPACE/startup/rootCA.pem"'

        assert 'SOURCE_WORKSPACE" == "/user_config"' in sub_start
        assert 'SOURCE_WORKSPACE" == /user_config/*' in sub_start
        assert 'SOURCE_WORKSPACE" == "/user_config"' in stop_fl
        assert 'SOURCE_WORKSPACE" == /user_config/*' in stop_fl
        assert 'WORKSPACE="/vault/workspace"' in sub_start
        assert 'WORKSPACE="/vault/workspace/$(basename "$SOURCE_WORKSPACE")"' in sub_start
        assert 'WORKSPACE="/vault/workspace"' in stop_fl
        assert 'WORKSPACE="/vault/workspace/$(basename "$SOURCE_WORKSPACE")"' in stop_fl
        assert 'KIT_WORKSPACE="$WORKSPACE/kit"' in sub_start
        assert copy_cmd in sub_start
        assert 'rm -rf "$KIT_WORKSPACE"' in sub_start
        assert 'rm -rf "$WORKSPACE"' not in sub_start
        assert 'ln -s "$KIT_WORKSPACE/startup" "$WORKSPACE/startup"' in sub_start
        assert 'ln -s "$KIT_WORKSPACE/local" "$WORKSPACE/local"' in sub_start
        assert '-m "$WORKSPACE"' in sub_start
        assert verify_cmd in sub_start
        assert sub_start.index(copy_cmd) < sub_start.index(verify_cmd)
        assert sub_start.index(verify_cmd) < sub_start.index('mkdir -p "$WORKSPACE/transfer"')
        assert 'touch "$WORKSPACE/shutdown.fl"' in stop_fl

    def test_master_template_class_allow_list_is_exact(self):
        """The provisioned allow list must match the curated component list exactly."""
        import os

        import yaml

        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "nvflare",
            "lighter",
            "templates",
            "master_template.yml",
        )
        assert os.path.exists(template_path)

        with open(template_path, "r") as f:
            template = yaml.safe_load(f)

        expected_paths = [
            "nvflare.app_common.aggregators.collect_and_assemble_model_aggregator.CollectAndAssembleModelAggregator",
            "nvflare.app_common.aggregators.intime_accumulate_model_aggregator.InTimeAccumulateWeightedAggregator",
            "nvflare.app_common.ccwf.CrossSiteEvalClientController",
            "nvflare.app_common.ccwf.CrossSiteEvalServerController",
            "nvflare.app_common.ccwf.CyclicClientController",
            "nvflare.app_common.ccwf.CyclicServerController",
            "nvflare.app_common.ccwf.SwarmClientController",
            "nvflare.app_common.ccwf.SwarmServerController",
            "nvflare.app_common.ccwf.comps.cwe_result_printer.CWEResultPrinter",
            "nvflare.app_common.ccwf.comps.np_file_model_persistor.NPFileModelPersistor",
            "nvflare.app_common.ccwf.comps.np_trainer.NPTrainer",
            "nvflare.app_common.ccwf.comps.simple_intime_model_selector.SimpleIntimeModelSelector",
            "nvflare.app_common.ccwf.comps.simple_model_shareable_generator.SimpleModelShareableGenerator",
            "nvflare.app_common.ccwf.cse_client_ctl.CrossSiteEvalClientController",
            "nvflare.app_common.ccwf.cse_server_ctl.CrossSiteEvalServerController",
            "nvflare.app_common.ccwf.cyclic_client_ctl.CyclicClientController",
            "nvflare.app_common.ccwf.cyclic_server_ctl.CyclicServerController",
            "nvflare.app_common.ccwf.swarm_client_ctl.SwarmClientController",
            "nvflare.app_common.ccwf.swarm_server_ctl.SwarmServerController",
            "nvflare.app_common.executors.statistics.statistics_executor.StatisticsExecutor",
            "nvflare.app_common.filters.exclude_vars.ExcludeVars",
            "nvflare.app_common.filters.percentile_privacy.PercentilePrivacy",
            "nvflare.app_common.filters.statistics_privacy_filter.StatisticsPrivacyFilter",
            "nvflare.app_common.filters.svt_privacy.SVTPrivacy",
            "nvflare.app_common.logging.job_log_receiver.JobLogReceiver",
            "nvflare.app_common.logging.job_log_streamer.JobLogStreamer",
            "nvflare.app_common.np.np_formatter.NPFormatter",
            "nvflare.app_common.np.np_model_locator.NPModelLocator",
            "nvflare.app_common.np.np_model_persistor.NPModelPersistor",
            "nvflare.app_common.np.np_trainer.NPTrainer",
            "nvflare.app_common.np.np_validator.NPValidator",
            "nvflare.app_common.psi.dh_psi.dh_psi_controller.DhPSIController",
            "nvflare.app_common.psi.file_psi_writer.FilePSIWriter",
            "nvflare.app_common.psi.psi_executor.PSIExecutor",
            "nvflare.app_common.shareablegenerators.full_model_shareable_generator.FullModelShareableGenerator",
            "nvflare.app_common.statistics.histogram_bins_cleanser.HistogramBinsCleanser",
            "nvflare.app_common.statistics.json_stats_file_persistor.JsonStatsFileWriter",
            "nvflare.app_common.statistics.min_count_cleanser.MinCountCleanser",
            "nvflare.app_common.statistics.min_max_cleanser.AddNoiseToMinMax",
            "nvflare.app_common.widgets.convert_to_fed_event.ConvertToFedEvent",
            "nvflare.app_common.widgets.event_recorder.ClientEventRecorder",
            "nvflare.app_common.widgets.event_recorder.ServerEventRecorder",
            "nvflare.app_common.widgets.intime_model_selector.IntimeModelSelector",
            "nvflare.app_common.widgets.validation_json_generator.ValidationJsonGenerator",
            "nvflare.app_common.workflows.cross_site_model_eval.CrossSiteModelEval",
            "nvflare.app_common.workflows.cyclic_ctl.CyclicController",
            "nvflare.app_common.workflows.fedavg.FedAvg",
            "nvflare.app_common.workflows.global_model_eval.GlobalModelEval",
            "nvflare.app_common.workflows.initialize_global_weights.InitializeGlobalWeights",
            "nvflare.app_common.workflows.lr.fedavg.FedAvgLR",
            "nvflare.app_common.workflows.lr.np_persistor.LRModelPersistor",
            "nvflare.app_common.workflows.scaffold.Scaffold",
            "nvflare.app_common.workflows.scatter_and_gather.ScatterAndGather",
            "nvflare.app_common.workflows.statistics_controller.StatisticsController",
            "nvflare.app_opt.he.intime_accumulate_model_aggregator.HEInTimeAccumulateWeightedAggregator",
            "nvflare.app_opt.he.model_decryptor.HEModelDecryptor",
            "nvflare.app_opt.he.model_encryptor.HEModelEncryptor",
            "nvflare.app_opt.he.model_serialize_filter.HEModelSerializeFilter",
            "nvflare.app_opt.he.model_shareable_generator.HEModelShareableGenerator",
            "nvflare.app_opt.psi.dh_psi.dh_psi_task_handler.DhPSITaskHandler",
            "nvflare.app_opt.pt.fedopt.PTFedOptModelShareableGenerator",
            "nvflare.app_opt.pt.file_model_locator.PTFileModelLocator",
            "nvflare.app_opt.pt.recipes.fedeval.EvalController",
            "nvflare.app_opt.sklearn.kmeans_assembler.KMeansAssembler",
            "nvflare.app_opt.sklearn.svm_assembler.SVMAssembler",
            "nvflare.app_opt.tf.fedopt_ctl.FedOpt",
            "nvflare.app_opt.tf.file_model_locator.TFFileModelLocator",
            "nvflare.app_opt.tracking.mlflow.mlflow_receiver.MLflowReceiver",
            "nvflare.app_opt.tracking.mlflow.mlflow_writer.MLflowWriter",
            "nvflare.app_opt.tracking.tb.tb_receiver.TBAnalyticsReceiver",
            "nvflare.app_opt.tracking.tb.tb_writer.TBWriter",
            "nvflare.app_opt.tracking.wandb.wandb_receiver.WandBReceiver",
            "nvflare.app_opt.xgboost.histogram_based.controller.XGBFedController",
            "nvflare.app_opt.xgboost.histogram_based.executor.FedXGBHistogramExecutor",
            "nvflare.app_opt.xgboost.histogram_based_v2.csv_data_loader.CSVDataLoader",
            "nvflare.app_opt.xgboost.histogram_based_v2.fed_controller.XGBFedController",
            "nvflare.app_opt.xgboost.histogram_based_v2.fed_executor.FedXGBHistogramExecutor",
            "nvflare.app_opt.xgboost.tree_based.bagging_aggregator.XGBBaggingAggregator",
            "nvflare.app_opt.xgboost.tree_based.executor.FedXGBTreeExecutor",
            "nvflare.app_opt.xgboost.tree_based.model_persistor.XGBModelPersistor",
            "nvflare.app_opt.xgboost.tree_based.shareable_generator.XGBModelShareableGenerator",
        ]
        assert expected_paths == list(DEFAULT_CLASS_ALLOW_LIST)
        for resource_key in ("local_client_resources", "local_server_resources"):
            resource_template = template[resource_key]
            assert _extract_class_allow_list(resource_template) == expected_paths
            assert '"class_list_enforcement_mode": "enforce"' in resource_template

    def test_master_template_class_allow_list_has_no_package_prefixes(self):
        """Package prefixes (entries ending in '.') broaden authorization to every class under that package.

        Future maintainers who add a broad prefix must enumerate the specific classes instead, or
        explicitly review-and-approve the prefix here. This guard prevents the previously-removed
        ``nvflare.edge.`` style entry from silently coming back via an expected_paths update.
        """
        import os

        import yaml

        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "nvflare",
            "lighter",
            "templates",
            "master_template.yml",
        )
        with open(template_path, "r") as f:
            template = yaml.safe_load(f)

        # If a future PR genuinely needs a broad package prefix, add it here with an explanation.
        explicitly_reviewed_package_prefixes: set = set()

        for resource_key in ("local_client_resources", "local_server_resources"):
            extracted = _extract_class_allow_list(template[resource_key])
            package_prefixes = {p for p in extracted if p.endswith(".")}
            unreviewed = package_prefixes - explicitly_reviewed_package_prefixes
            assert not unreviewed, (
                f"unreviewed package prefixes in {resource_key}: {sorted(unreviewed)}. "
                "Package prefixes authorize every class (current and future) under the package, "
                "which broadens the security posture of the allow_list. Either enumerate the "
                "specific classes needed, or add the prefix to "
                "explicitly_reviewed_package_prefixes with an in-test explanation."
            )

    def test_master_template_class_allow_list_excludes_edge_components(self):
        """Provisioned non-BYOC resources must not authorize edge components."""
        import os

        import yaml

        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "nvflare",
            "lighter",
            "templates",
            "master_template.yml",
        )
        with open(template_path, "r") as f:
            template = yaml.safe_load(f)

        for resource_key in ("local_client_resources", "local_server_resources"):
            extracted = _extract_class_allow_list(template[resource_key])
            edge_paths = [p for p in extracted if p.startswith("nvflare.edge.")]
            assert not edge_paths, f"edge classes should not be in {resource_key}: {edge_paths}"

    def test_master_template_default_authz_grants_submission_byoc_to_lead_and_project_admin(self):
        """Default authorization grants broad project_admin permissions and lead BYOC submission permission."""
        import os

        import yaml

        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "nvflare",
            "lighter",
            "templates",
            "master_template.yml",
        )
        with open(template_path, "r") as f:
            template = yaml.safe_load(f)

        default_authz = yaml.safe_load(template["default_authz"])
        permissions = default_authz["permissions"]

        assert permissions["project_admin"] == "any"
        assert permissions["lead"]["byoc"] == "any"

    def test_master_template_allows_non_edge_regression_components(self):
        """Pin the specific paths that regressed in 2.8.0rc4 (PR #4701 fallout).

        Non-BYOC jobs that loaded these built-in classes worked on 2.8.0rc3
        and broke on rc4 because they were missing from the provisioned
        class_allow_list. Edge components are intentionally outside this
        non-BYOC regression set.
        """
        import os

        import yaml

        from nvflare.app_common.widgets.component_path_authorizer import ComponentPathAuthorizer

        template_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "nvflare",
            "lighter",
            "templates",
            "master_template.yml",
        )
        with open(template_path, "r") as f:
            template = yaml.safe_load(f)

        regression_paths = [
            # PR #4701 regression: NPTrainer was omitted from the curated list.
            "nvflare.app_common.np.np_trainer.NPTrainer",
            # PR #4701 regression: ccwf re-export short paths were not matched
            # because the list only stored the full module path. Configs in
            # the wild reference the package-level alias.
            "nvflare.app_common.ccwf.CyclicServerController",
            "nvflare.app_common.ccwf.SwarmClientController",
            "nvflare.app_common.ccwf.CrossSiteEvalServerController",
        ]
        for resource_key in ("local_client_resources", "local_server_resources"):
            allow_list = _extract_class_allow_list(template[resource_key])
            for path in regression_paths:
                assert any(
                    ComponentPathAuthorizer._path_matches_prefix(path, prefix) for prefix in allow_list
                ), f"{path!r} would be rejected by ComponentPathAuthorizer against {resource_key}"
