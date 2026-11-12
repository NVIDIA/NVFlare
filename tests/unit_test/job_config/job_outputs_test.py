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

import importlib
import os
import re

import pytest

from nvflare.apis.job_def import MAX_JOB_OUTPUTS
from nvflare.apis.job_outputs import (
    as_folder,
    get_output_resolution,
    has_resolver_ancestor,
    in_server_app,
    register_output_resolver,
)
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.np.np_model_persistor import NPModelPersistor
from nvflare.app_common.widgets.metrics_artifact_writer import MetricsArtifactWriter
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.job_config.fed_app_config import ServerAppConfig
from nvflare.job_config.job_outputs import outputs_from, resolve_job_outputs


class _Writer:
    def __init__(self, name="out.json"):
        self.name = name


register_output_resolver(_Writer, lambda w: [w.name])


class _SubWriter(_Writer):
    pass


def test_registry_resolution_is_by_exact_class():
    resolution = get_output_resolution(_Writer("a.json"))

    assert resolution.resolver(_Writer("a.json")) == ["a.json"]
    assert resolution.supporting is False
    assert get_output_resolution(_SubWriter()) is None  # a subclass may write elsewhere


def test_registry_a_subclass_of_a_resolved_class_is_known_to_have_one_above_it():
    assert has_resolver_ancestor(_SubWriter())
    assert not has_resolver_ancestor(_Writer())  # its own class, not an ancestor
    assert not has_resolver_ancestor(object())


def test_registry_supporting_resolvers_are_marked():
    class _Metrics:
        pass

    register_output_resolver(_Metrics, lambda m: ["metrics/"], supporting=True)

    assert get_output_resolution(_Metrics()).supporting is True


@pytest.mark.parametrize("component_class, resolver", [(_Writer(), lambda w: []), (_Writer, "not callable")])
def test_registry_register_rejects_a_non_class_or_a_non_callable(component_class, resolver):
    with pytest.raises(TypeError):
        register_output_resolver(component_class, resolver)


@pytest.mark.parametrize("name, expected", [("m.pt", "app_server/m.pt"), ("", None), (None, None)])
def test_registry_in_server_app(name, expected):
    assert in_server_app(name) == expected


@pytest.mark.parametrize("path, expected", [("metrics", "metrics/"), ("./m/", "m/"), (".", None), ("", None)])
def test_registry_as_folder(path, expected):
    assert as_folder(path) == expected


class _OwnPersistor(ModelPersistor):
    """A user's own persistor: nothing says where it saves the model."""

    def load_model(self, fl_ctx):
        return None

    def save_model(self, model, fl_ctx):
        pass


def _server(*components, workflows=()):
    app = ServerAppConfig()
    for i, component in enumerate(components):
        app.add_component(f"c{i}", component)
    for i, controller in enumerate(workflows):
        app.add_workflow(f"w{i}", controller)
    return app


def _paths(component):
    resolution = get_output_resolution(component)
    assert resolution is not None, f"{type(component).__name__} has no output resolver"
    return [p for p in resolution.resolver(component) if p]


class TestResolvers:
    def test_pytorch_persistor_names_its_files_from_its_settings(self):
        persistor = PTFileModelPersistor(global_model_file_name="m.pt", best_global_model_file_name="best.pt")

        assert _paths(persistor) == ["app_server/m.pt", "app_server/best.pt"]

    @pytest.mark.parametrize(
        "persistor, expected",
        [
            (NPModelPersistor(), ["models/server.npy"]),
            (NPModelPersistor(model_dir="out", model_name="final"), ["out/final.npy"]),  # np.save adds .npy
        ],
    )
    def test_numpy_persistor_names_its_npy_file(self, persistor, expected):
        assert _paths(persistor) == expected

    def test_logistic_regression_persistor_names_its_npy_file(self):
        from nvflare.app_common.workflows.lr.np_persistor import LRModelPersistor

        assert _paths(LRModelPersistor(n_features=3)) == ["models/weights.npy"]

    @pytest.mark.parametrize(
        "module, cls, expected",
        [
            ("nvflare.app_opt.sklearn.joblib_model_param_persistor", "JoblibModelParamPersistor", "model_param.joblib"),
            ("nvflare.app_opt.xgboost.tree_based.model_persistor", "XGBModelPersistor", "xgboost_model.json"),
            ("nvflare.app_opt.tf.model_persistor", "TFModelPersistor", "tf_model.weights.h5"),
        ],
    )
    def test_framework_persistors_name_their_file_in_the_server_app(self, module, cls, expected):
        persistor_class = getattr(pytest.importorskip(module), cls)
        persistor = persistor_class() if cls != "TFModelPersistor" else persistor_class(model=None)

        assert _paths(persistor) == [f"app_server/{expected}"]

    def test_disk_backed_fedavg_persistor_names_its_saved_model(self):
        from nvflare.app_opt.pt.disk_fedavg_components import DiskFedAvgPersistor

        assert _paths(DiskFedAvgPersistor("init.safetensors")) == ["app_server/FL_global_model.safetensors"]

    def test_fedavg_names_the_model_it_saves_only_without_a_persistor(self):
        assert _paths(FedAvg(num_clients=1, num_rounds=1, persistor_id="", save_filename="final.pt")) == ["final.pt"]
        assert _paths(FedAvg(num_clients=1, num_rounds=1)) == []  # its persistor saves the model

    def test_cross_site_evaluation_names_the_folder_it_was_given(self):
        assert _paths(CrossSiteModelEval()) == ["cross_site_val/"]
        assert _paths(CrossSiteModelEval(cross_val_dir="my_eval")) == ["my_eval/"]

    @pytest.mark.parametrize(
        "writer, expected",
        [
            (MetricsArtifactWriter(), ["metrics/"]),
            (MetricsArtifactWriter(results_dir="./my_metrics/"), ["my_metrics/"]),
            (MetricsArtifactWriter(results_dir="."), ["metrics_summary.json", "round_metrics.jsonl"]),
            (MetricsArtifactWriter(write_artifacts=False), []),
        ],
    )
    def test_metrics_writer_names_its_folder_or_its_files_as_supporting(self, writer, expected):
        assert _paths(writer) == expected
        assert get_output_resolution(writer).supporting is True

    def test_statistics_writer_names_its_output_file(self):
        from nvflare.app_common.statistics.json_stats_file_persistor import JsonStatsFileWriter

        assert _paths(JsonStatsFileWriter(output_path="statistics/adults.json")) == ["statistics/adults.json"]


class TestResolveJobOutputs:
    def test_collects_main_and_supporting_outputs(self):
        server = _server(PTFileModelPersistor(), MetricsArtifactWriter(), workflows=[CrossSiteModelEval()])

        assert resolve_job_outputs(server) == [
            "app_server/FL_global_model.pt",
            "app_server/best_FL_global_model.pt",
            "cross_site_val/",
            "metrics/",  # supporting outputs come after the main ones
        ]

    def test_a_setting_changed_after_the_component_was_added_is_used(self):
        persistor = PTFileModelPersistor()
        server = _server(persistor)
        persistor.global_model_file_name = "renamed.pt"

        assert "app_server/renamed.pt" in resolve_job_outputs(server)

    def test_a_persistor_without_a_resolver_declares_nothing(self):
        assert resolve_job_outputs(_server(PTFileModelPersistor(), _OwnPersistor())) == []

    @pytest.mark.parametrize("base", [PTFileModelPersistor, MetricsArtifactWriter])
    def test_a_subclass_of_a_resolved_class_declares_nothing(self, base):
        subclass = type("_Elsewhere", (base,), {})  # may write somewhere its base does not

        assert resolve_job_outputs(_server(NPModelPersistor(), subclass())) == []

    def test_a_users_own_persistor_can_register_its_files(self):
        class _Registered(_OwnPersistor):
            def __init__(self, path):
                super().__init__()
                self.path = path

        register_output_resolver(_Registered, lambda p: [p.path])

        assert resolve_job_outputs(_server(_Registered("models/final.bin"))) == ["models/final.bin"]

    def test_components_that_write_no_results_are_skipped(self):
        from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator

        server = _server(NPModelPersistor(), InTimeAccumulateWeightedAggregator(expected_data_kind="WEIGHTS"))

        assert resolve_job_outputs(server) == ["models/server.npy"]

    def test_a_failing_resolver_declares_nothing_rather_than_failing_the_export(self):
        class _Broken(_OwnPersistor):
            pass

        register_output_resolver(_Broken, lambda p: p.missing_attribute)

        assert resolve_job_outputs(_server(NPModelPersistor(), _Broken())) == []

    def test_only_supporting_outputs_declare_nothing(self):
        # metrics, and a controller saving a model nothing names: the main result is unknown
        assert resolve_job_outputs(_server(MetricsArtifactWriter())) == []

    def test_a_rejected_path_declares_nothing(self):
        server = _server(NPModelPersistor(model_dir="/data/out"), MetricsArtifactWriter())

        assert resolve_job_outputs(server) == []

    @pytest.mark.parametrize("dest_dir", [None, ".", "./"])
    @pytest.mark.parametrize("as_folder_source", [False, True])
    def test_a_shipped_server_environment_declares_nothing(self, tmp_path, as_folder_source, dest_dir):
        # environment.json can move where components write, such as PTFileModelPersistor's checkpoint folder
        config = tmp_path / "server_config"
        config.mkdir()
        (config / "environment.json").write_text("{}")
        server = _server(PTFileModelPersistor())
        source = config if as_folder_source else config / "environment.json"
        server.add_file_source(str(source), dest_dir, "config")

        assert resolve_job_outputs(server) == []

    def test_an_environment_file_in_a_config_subfolder_is_not_the_apps(self, tmp_path):
        env_file = tmp_path / "environment.json"
        env_file.write_text("{}")
        server = _server(PTFileModelPersistor())
        server.add_file_source(str(env_file), "extra", "config")

        assert "app_server/FL_global_model.pt" in resolve_job_outputs(server)


class TestOutputsFrom:
    def test_normalizes_and_removes_duplicates_and_covered_files(self):
        paths = ["./output", "metrics/", "metrics/metrics_summary.json", "output", "app_server/FL_global_model.pt"]

        assert outputs_from(paths) == ["output", "metrics/", "app_server/FL_global_model.pt"]

    @pytest.mark.parametrize("path", ["../outside.json", ".", "/abs/stats.json", ""])
    def test_an_undeclarable_path_declares_nothing(self, path):
        assert outputs_from([path, "metrics/"]) == []

    def test_more_outputs_than_allowed_declare_nothing(self):
        assert outputs_from([f"stats/{i}.json" for i in range(MAX_JOB_OUTPUTS + 1)]) == []


# Built-in model persistors that declare nothing on purpose: a job using one downloads whole.
_UNRESOLVED_PERSISTORS = {
    "AppDefinedModelPersistor",  # an abstract base: an application's own subclass decides where it saves
    "NPFileModelPersistor",  # client-controlled workflows run it on the clients, not the server
}


def _builtin_model_persistors():
    """(module, class name) of every ModelPersistor subclass defined in nvflare, found from the source."""
    root = os.path.dirname(importlib.import_module("nvflare").__file__)
    persistor_names = {"ModelPersistor"}
    found = {}
    sources = []
    for folder, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(folder, name)
                with open(path, encoding="utf-8") as f:
                    sources.append((path, f.read()))
    class_re = re.compile(r"^class (\w+)\(([^)]*)\)", re.MULTILINE)
    changed = True
    while changed:  # follow subclasses of subclasses
        changed = False
        for path, text in sources:
            for cls, bases in class_re.findall(text):
                if cls not in persistor_names and any(
                    b.strip().split(".")[-1] in persistor_names for b in bases.split(",")
                ):
                    persistor_names.add(cls)
                    module = os.path.relpath(path, os.path.dirname(root))[:-3].replace(os.sep, ".")
                    found[cls] = module
                    changed = True
    return sorted(found.items())


@pytest.mark.parametrize("cls, module", _builtin_model_persistors())
def test_every_builtin_model_persistor_names_its_files_or_is_listed(cls, module):
    if cls in _UNRESOLVED_PERSISTORS:
        return
    persistor_class = getattr(pytest.importorskip(module), cls)
    from nvflare.apis.job_outputs import _RESOLVERS

    assert persistor_class in _RESOLVERS, (
        f"{module}.{cls} saves a model but registers no output resolver; register one beside the class, or add it "
        "to _UNRESOLVED_PERSISTORS if its files cannot be named"
    )
