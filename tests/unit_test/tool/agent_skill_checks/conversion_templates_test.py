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

"""Deterministic tests for the packaged conversion templates.

These run the shipped PyTorch, Lightning, and Hugging Face client templates
plus the custom ``ModelAggregator`` template against toy models so template rot
is caught here rather than only in expensive, nondeterministic LLM evals.
"""

import ast
import importlib.util
import inspect
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SKILLS_ROOT = REPO_ROOT / "skills"
PT_TEMPLATES = SKILLS_ROOT / "nvflare-convert-pytorch" / "assets"
LIGHTNING_TEMPLATES = SKILLS_ROOT / "nvflare-convert-lightning" / "assets"
HF_TEMPLATES = SKILLS_ROOT / "nvflare-convert-huggingface" / "assets"
SHARED_TEMPLATES = SKILLS_ROOT / "nvflare-shared" / "assets"


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"template_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FloatOverflow:
    def __float__(self):
        raise OverflowError("step count too large")


def test_non_fedavg_tensor_profile_omits_unsupported_disk_offload(tmp_path):
    pytest.importorskip("torch")
    from nvflare.app_opt.pt.recipes.cyclic import CyclicRecipe
    from nvflare.client.config import ExchangeFormat

    train_script = tmp_path / "client.py"
    train_script.write_text("pass\n", encoding="utf-8")
    parameters = inspect.signature(CyclicRecipe).parameters

    assert "server_expected_format" in parameters
    assert "enable_tensor_disk_offload" not in parameters

    recipe = CyclicRecipe(
        name="skill-capability-test",
        model={"class_path": "torch.nn.Linear", "args": {"in_features": 2, "out_features": 1}},
        train_script=str(train_script),
        min_clients=2,
        server_expected_format=ExchangeFormat.PYTORCH,
    )
    recipe.add_decomposers(["nvflare.app_opt.pt.decomposers.TensorDecomposer"])

    assert recipe.server_expected_format == ExchangeFormat.PYTORCH


def test_pytorch_eval_template_computes_metric_against_toy_model():
    torch = pytest.importorskip("torch")
    module = _load_module(PT_TEMPLATES / "client_with_eval.py")

    model = torch.nn.Linear(4, 2)
    features = torch.randn(6, 4)
    labels = torch.randint(0, 2, (6,))
    val_loader = [(features, labels)]

    metric = module.evaluate(model, val_loader, device="cpu")

    assert isinstance(metric, float)
    assert 0.0 <= metric <= 1.0


def test_pytorch_eval_template_restores_training_mode():
    # evaluate() must not leave the model in eval mode, or a later training round
    # would run with dropout/batchnorm disabled.
    torch = pytest.importorskip("torch")
    module = _load_module(PT_TEMPLATES / "client_with_eval.py")

    model = torch.nn.Linear(4, 2)
    model.train()
    features = torch.randn(4, 4)
    labels = torch.randint(0, 2, (4,))

    module.evaluate(model, [(features, labels)], device="cpu")

    assert model.training is True


def test_pytorch_eval_template_fails_closed_on_empty_data():
    pytest.importorskip("torch")
    module = _load_module(PT_TEMPLATES / "client_with_eval.py")

    with pytest.raises(RuntimeError):
        module.evaluate(_DummyModel(), [], device="cpu")


def test_pytorch_eval_template_initializes_flare_before_training_setup():
    source = (PT_TEMPLATES / "client_with_eval.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main_func = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")

    init_line = None
    setup_line = None
    loop_line = None
    for node in ast.walk(main_func):
        if isinstance(node, ast.While):
            loop_line = node.lineno
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "flare"
            and node.func.attr == "init"
        ):
            init_line = node.lineno
        if isinstance(node.func, ast.Name) and node.func.id == "train_setup_factory":
            setup_line = node.lineno

    assert init_line is not None
    assert setup_line is not None
    assert loop_line is not None
    assert init_line < setup_line < loop_line


@pytest.mark.parametrize(
    "evaluate_before_train, expected_events",
    [
        (True, ["init", "factory", "patch", "is_running", "evaluate", "train", "is_running"]),
        (False, ["init", "factory", "patch", "is_running", "train", "is_running"]),
    ],
)
def test_huggingface_client_template_preserves_trainer_sequence(monkeypatch, evaluate_before_train, expected_events):
    module = _load_module(HF_TEMPLATES / "client_with_eval.py")
    events = []

    class _Trainer:
        def evaluate(self):
            events.append("evaluate")

        def train(self):
            events.append("train")

    def trainer_factory():
        events.append("factory")
        return _Trainer()

    running = iter((True, False))
    monkeypatch.setattr(module.flare, "init", lambda rank=0: events.append("init"))
    monkeypatch.setattr(module.flare, "patch", lambda trainer: events.append("patch"))
    monkeypatch.setattr(
        module.flare,
        "is_running",
        lambda: events.append("is_running") or next(running),
    )

    module.main(trainer_factory, evaluate_before_train=evaluate_before_train)

    assert events == expected_events


def test_huggingface_client_template_has_one_patch_and_no_manual_exchange():
    source = (HF_TEMPLATES / "client_with_eval.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "flare"
    ]

    assert [node.func.attr for node in calls].count("patch") == 1
    assert not {node.func.attr for node in calls} & {"receive", "send"}


def test_huggingface_server_model_template_returns_source_model_without_wrapper_prefix(monkeypatch):
    torch = pytest.importorskip("torch")
    expected_model = torch.nn.Linear(4, 2)
    source_model_module = types.ModuleType("model")
    source_model_module.load_model = lambda model_name_or_path, **kwargs: expected_model
    monkeypatch.setitem(sys.modules, "model", source_model_module)

    module = _load_module(HF_TEMPLATES / "server_model.py")
    server_model = module.ServerModel("local-model")

    assert server_model is expected_model
    assert set(server_model.state_dict()) == {"weight", "bias"}


def test_huggingface_job_template_uses_pytorch_fast_path_and_packages_model_files(monkeypatch):
    module = _load_module(HF_TEMPLATES / "job.py")

    class _Recipe:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.server_files = []
            self.client_files = []

        def add_server_file(self, path):
            self.server_files.append(path)

        def add_client_file(self, path):
            self.client_files.append(path)

    monkeypatch.setattr(module, "FedAvgRecipe", _Recipe)
    recipe = module.build_recipe(
        name="hf-test",
        model_name_or_path="local-model",
        data_root="/tmp/data",
        num_clients=2,
        num_rounds=3,
        key_metric="eval_accuracy",
    )

    assert recipe.kwargs["model"] == {
        "class_path": "server_model.ServerModel",
        "args": {"model_name_or_path": "local-model"},
    }
    assert recipe.kwargs["train_script"] == "client.py"
    assert recipe.kwargs["min_clients"] == 2
    assert recipe.kwargs["num_rounds"] == 3
    assert recipe.kwargs["key_metric"] == "eval_accuracy"
    assert [Path(path).name for path in recipe.server_files] == ["server_model.py", "model.py"]
    assert [Path(path).name for path in recipe.client_files] == ["client.py", "model.py"]
    assert "--max_steps 10" in recipe.kwargs["train_args"]
    assert "--num_train_epochs" not in recipe.kwargs["train_args"]


def test_huggingface_job_template_supports_one_resolved_budget_mode():
    module = _load_module(HF_TEMPLATES / "job.py")

    requested_steps = module.build_train_args("local-model", "/tmp/data", 2, max_steps=7)
    requested_epochs = module.build_train_args("local-model", "/tmp/data", 2, num_train_epochs=3.0)
    preserved_source = module.build_train_args("local-model", "/tmp/data", 2, preserve_source_budget=True)

    assert "--max_steps 7" in requested_steps
    assert "--num_train_epochs" not in requested_steps
    assert "--num_train_epochs 3.0" in requested_epochs
    assert "--max_steps" not in requested_epochs
    assert "--max_steps" not in preserved_source
    assert "--num_train_epochs" not in preserved_source

    with pytest.raises(ValueError, match="only one"):
        module.build_train_args("local-model", "/tmp/data", 2, max_steps=7, num_train_epochs=3.0)


@pytest.mark.parametrize("max_steps", [True, 0, -1, 1.5])
def test_huggingface_job_template_rejects_invalid_programmatic_step_budgets(max_steps):
    module = _load_module(HF_TEMPLATES / "job.py")

    with pytest.raises(ValueError, match="positive integer"):
        module.build_train_args("local-model", "/tmp/data", 2, max_steps=max_steps)


@pytest.mark.parametrize("num_train_epochs", [True, 0, -1, float("nan"), float("inf")])
def test_huggingface_job_template_rejects_invalid_programmatic_epoch_budgets(num_train_epochs):
    module = _load_module(HF_TEMPLATES / "job.py")

    with pytest.raises(ValueError, match="finite positive number"):
        module.build_train_args("local-model", "/tmp/data", 2, num_train_epochs=num_train_epochs)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--max_steps", "0"),
        ("--max_steps", "-1"),
        ("--max_steps", "1.5"),
        ("--num_train_epochs", "0"),
        ("--num_train_epochs", "-1"),
        ("--num_train_epochs", "nan"),
        ("--num_train_epochs", "inf"),
    ],
)
def test_huggingface_job_template_cli_rejects_invalid_budgets(monkeypatch, option, value):
    module = _load_module(HF_TEMPLATES / "job.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "job.py",
            "--model_name_or_path",
            "local-model",
            "--data_root",
            "/tmp/data",
            "--key_metric",
            "eval_accuracy",
            option,
            value,
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        module.main()


def test_huggingface_job_template_rejects_unrepresentable_in_process_arguments():
    module = _load_module(HF_TEMPLATES / "job.py")

    with pytest.raises(ValueError, match="whitespace-free"):
        module.build_train_args("model with spaces", "/tmp/data", 2)


def test_huggingface_job_template_exports_colocated_files_from_another_working_directory(tmp_path, monkeypatch):
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    job_path = generated_dir / "job.py"
    job_path.write_text((HF_TEMPLATES / "job.py").read_text(encoding="utf-8"), encoding="utf-8")
    (generated_dir / "client.py").write_text("import model\n", encoding="utf-8")
    (generated_dir / "model.py").write_text("class Model:\n    pass\n", encoding="utf-8")
    (generated_dir / "server_model.py").write_text("from model import Model\n", encoding="utf-8")

    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    monkeypatch.chdir(caller_dir)
    module = _load_module(job_path)
    recipe = module.build_recipe(
        name="hf-cwd-independent",
        model_name_or_path="local-model",
        data_root="/tmp/data",
        num_clients=2,
        num_rounds=1,
        key_metric="eval_accuracy",
    )
    assert Path.cwd() == caller_dir
    export_root = tmp_path / "export"
    recipe.export(str(export_root))

    app_dir = export_root / recipe.name / "app"
    assert (app_dir / "custom" / "client.py").is_file()
    assert (app_dir / "custom" / "server_model.py").is_file()
    assert (app_dir / "custom" / "model.py").is_file()
    client_config = json.loads((app_dir / "config" / "config_fed_client.json").read_text(encoding="utf-8"))
    executor_args = client_config["executors"][0]["executor"]["args"]
    assert executor_args["task_script_path"] == "client.py"


def test_huggingface_job_template_exports_per_site_files_from_another_working_directory(tmp_path, monkeypatch):
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    job_path = generated_dir / "job.py"
    job_path.write_text((HF_TEMPLATES / "job.py").read_text(encoding="utf-8"), encoding="utf-8")
    (generated_dir / "client.py").write_text("import model\n", encoding="utf-8")
    (generated_dir / "model.py").write_text("class Model:\n    pass\n", encoding="utf-8")
    (generated_dir / "server_model.py").write_text("from model import Model\n", encoding="utf-8")

    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    monkeypatch.chdir(caller_dir)
    module = _load_module(job_path)
    site_args = {
        "site-1": {"train_args": module.build_train_args("local-model", "/data/site-1", 2, max_steps=2)},
        "site-2": {"train_args": module.build_train_args("local-model", "/data/site-2", 2, max_steps=3)},
    }
    recipe = module.build_recipe(
        name="hf-per-site-cwd-independent",
        model_name_or_path="local-model",
        data_root="/tmp/data",
        num_clients=2,
        num_rounds=1,
        key_metric="eval_accuracy",
        per_site_config=site_args,
    )
    assert Path.cwd() == caller_dir
    export_root = tmp_path / "export"
    recipe.export(str(export_root))

    job_root = export_root / recipe.name
    assert (job_root / "app_server" / "custom" / "server_model.py").is_file()
    assert (job_root / "app_server" / "custom" / "model.py").is_file()
    for site_name, expected in site_args.items():
        app_dir = job_root / f"app_{site_name}"
        assert (app_dir / "custom" / "client.py").is_file()
        assert (app_dir / "custom" / "model.py").is_file()
        client_config = json.loads((app_dir / "config" / "config_fed_client.json").read_text(encoding="utf-8"))
        executor_args = client_config["executors"][0]["executor"]["args"]
        assert executor_args["task_script_path"] == "client.py"
        assert executor_args["task_script_args"] == expected["train_args"]


def test_huggingface_job_template_rejects_deprecated_per_site_constructor_option():
    module = _load_module(HF_TEMPLATES / "job.py")

    with pytest.raises(ValueError, match="pass per_site_config to build_recipe"):
        module.build_recipe(
            name="hf-deprecated-per-site",
            model_name_or_path="local-model",
            data_root="/tmp/data",
            num_clients=2,
            num_rounds=1,
            key_metric="eval_accuracy",
            recipe_options={"per_site_config": {"site-1": {}, "site-2": {}}},
        )


@pytest.mark.parametrize(
    "train_script",
    ["client_site_2.py", str(HF_TEMPLATES / "client_with_eval.py")],
)
def test_huggingface_job_template_rejects_per_site_train_script_overrides(tmp_path, monkeypatch, train_script):
    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    monkeypatch.chdir(caller_dir)
    module = _load_module(HF_TEMPLATES / "job.py")

    with pytest.raises(ValueError, match="must use the shared train_script='client.py'"):
        module.build_recipe(
            name="hf-per-site-script",
            model_name_or_path="local-model",
            data_root="/tmp/data",
            num_clients=2,
            num_rounds=1,
            key_metric="eval_accuracy",
            per_site_config={"site-1": {}, "site-2": {"train_script": train_script}},
        )

    assert Path.cwd() == caller_dir


def test_huggingface_job_template_restores_cwd_when_per_site_config_is_invalid(tmp_path, monkeypatch):
    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    monkeypatch.chdir(caller_dir)
    module = _load_module(HF_TEMPLATES / "job.py")

    with pytest.raises(ValueError, match="min_clients=2"):
        module.build_recipe(
            name="hf-invalid-per-site",
            model_name_or_path="local-model",
            data_root="/tmp/data",
            num_clients=2,
            num_rounds=1,
            key_metric="eval_accuracy",
            per_site_config={"site-1": {}},
        )

    assert Path.cwd() == caller_dir


def test_huggingface_client_template_rejects_abbreviated_hf_arguments():
    transformers = pytest.importorskip("transformers")

    @dataclass
    class ProjectArguments:
        max_train_samples: int | None = None

    module = _load_module(HF_TEMPLATES / "client_with_eval.py")
    parser = module.make_hf_argument_parser((ProjectArguments, transformers.TrainingArguments))

    parsed, _ = parser.parse_args_into_dataclasses(args=["--output_dir", "/tmp/output", "--max_train_samples", "7"])
    assert parsed.max_train_samples == 7

    with pytest.raises(ValueError, match="max_train_samp"):
        parser.parse_args_into_dataclasses(args=["--output_dir", "/tmp/output", "--max_train_samp", "7"])


def test_huggingface_job_template_uses_public_recipe_execution_without_internal_probes():
    source = (HF_TEMPLATES / "job.py").read_text(encoding="utf-8")

    assert "FedAvgRecipe(" in source
    assert "SimEnv(" in source
    assert "recipe.execute(" in source
    assert "from nvflare.recipe import SimEnv" in source
    assert "inspect." not in source
    assert "PTModel" not in source
    assert "persistor" not in source.lower()


def test_custom_aggregator_template_step_weighted_average():
    import numpy as np

    from nvflare.apis.dxo import MetaKey
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    aggregator.accept_model(FLModel(params={"w": np.array([2.0])}, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1}))
    aggregator.accept_model(FLModel(params={"w": np.array([4.0])}, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: "3"}))
    result = aggregator.aggregate_model()

    # (2*1 + 4*3) / (1 + 3) = 14 / 4 = 3.5
    assert result.params["w"][0] == pytest.approx(3.5)


def test_custom_aggregator_template_carries_weighted_metrics():
    import numpy as np

    from nvflare.apis.dxo import MetaKey
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    aggregator.accept_model(
        FLModel(
            params={"w": np.array([2.0])},
            metrics={
                "accuracy": 0.5,
                "loss": 2.0,
                "nan_metric": float("nan"),
                "flag": True,
                "numeric_string": "0.95",
            },
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1},
        )
    )
    aggregator.accept_model(
        FLModel(
            params={"w": np.array([4.0])},
            metrics={"accuracy": 0.75, "loss": 1.0},
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 3},
        )
    )
    result = aggregator.aggregate_model()

    assert result.metrics["accuracy"] == pytest.approx((0.5 * 1 + 0.75 * 3) / 4)
    assert result.metrics["loss"] == pytest.approx((2.0 * 1 + 1.0 * 3) / 4)
    assert result.metrics["flag"] == pytest.approx(1.0)
    assert "nan_metric" not in result.metrics
    assert "numeric_string" not in result.metrics


def test_custom_aggregator_template_disables_metrics_when_any_client_omits_them():
    import numpy as np

    from nvflare.apis.dxo import MetaKey
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    aggregator.accept_model(
        FLModel(
            params={"w": np.array([2.0])},
            metrics={"accuracy": 0.5},
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1},
        )
    )
    aggregator.accept_model(
        FLModel(params={"w": np.array([4.0])}, metrics=None, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 3})
    )
    result = aggregator.aggregate_model()

    assert result.params["w"][0] == pytest.approx(3.5)
    assert result.metrics is None


def test_custom_aggregator_template_uses_per_key_metric_denominators():
    import numpy as np

    from nvflare.apis.dxo import MetaKey
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    aggregator.accept_model(
        FLModel(
            params={"w": np.array([2.0])},
            metrics={"accuracy": 0.5, "loss": 2.0},
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1},
        )
    )
    aggregator.accept_model(
        FLModel(
            params={"w": np.array([4.0])},
            metrics={"accuracy": 0.75},
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 3},
        )
    )
    result = aggregator.aggregate_model()

    assert result.metrics["accuracy"] == pytest.approx((0.5 * 1 + 0.75 * 3) / 4)
    assert result.metrics["loss"] == pytest.approx(2.0)


def test_custom_aggregator_template_keeps_extreme_weighted_metrics_finite():
    import math

    import numpy as np

    from nvflare.apis.dxo import MetaKey
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    for score in (1e308, -1e308):
        aggregator.accept_model(
            FLModel(
                params={"w": np.array([0.0])},
                metrics={"score": score},
                meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1e308},
            )
        )
    result = aggregator.aggregate_model()

    assert math.isfinite(result.metrics["score"])
    assert result.metrics["score"] == pytest.approx(0.0)


def test_custom_aggregator_template_keeps_extreme_weighted_params_finite():
    import math

    import numpy as np

    from nvflare.apis.dxo import MetaKey
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    for value in (0.5, 0.9):
        aggregator.accept_model(
            FLModel(
                params={"w": np.array([value])},
                meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1e308},
            )
        )
    result = aggregator.aggregate_model()

    assert math.isfinite(result.params["w"][0])
    assert result.params["w"][0] == pytest.approx(0.7)


def test_custom_aggregator_template_materializes_lazy_disk_offload_refs():
    # With enable_tensor_disk_offload=True, params can arrive as lazy references
    # exposing materialize() instead of in-memory arrays. The template must
    # materialize before the weighted-sum math rather than doing value * weight
    # on the ref (which would raise TypeError).
    import numpy as np

    from nvflare.apis.dxo import MetaKey
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    class _LazyRef:
        def __init__(self, array):
            self._array = array

        def materialize(self):
            return self._array

        def __mul__(self, other):  # pragma: no cover - must never be reached
            raise TypeError("lazy ref must be materialized before weighted math")

    aggregator.accept_model(FLModel(params={"w": _LazyRef(np.array([2.0]))}, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1}))
    aggregator.accept_model(FLModel(params={"w": _LazyRef(np.array([4.0]))}, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 3}))
    result = aggregator.aggregate_model()

    # (2*1 + 4*3) / (1 + 3) = 3.5, computed on the materialized arrays.
    assert result.params["w"][0] == pytest.approx(3.5)


def test_custom_aggregator_template_averages_per_key_with_mismatched_keys():
    # A parameter present in only one client is averaged over just that client's
    # weight (not diluted), and a key missing from the first client does not
    # raise KeyError.
    import numpy as np

    from nvflare.apis.dxo import MetaKey
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    aggregator.accept_model(FLModel(params={"shared": np.array([2.0])}, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 1}))
    aggregator.accept_model(
        FLModel(
            params={"shared": np.array([4.0]), "only_b": np.array([9.0])},
            meta={MetaKey.NUM_STEPS_CURRENT_ROUND: 3},
        )
    )
    result = aggregator.aggregate_model()

    # shared: (2*1 + 4*3)/(1+3) = 3.5 ; only_b: 9 present only in client B -> 9.0
    assert result.params["shared"][0] == pytest.approx(3.5)
    assert result.params["only_b"][0] == pytest.approx(9.0)


@pytest.mark.parametrize(
    "bad_steps",
    [
        -5,
        0,
        float("nan"),
        float("inf"),
        "abc",
        True,
        None,
        pytest.param(10**10000, id="oversized-int"),
        pytest.param(_FloatOverflow(), id="overflow"),
    ],
)
def test_custom_aggregator_template_falls_back_to_unit_weight_for_bad_step_counts(bad_steps):
    # Negative / non-finite / non-numeric / bool / missing / overflowing step
    # metadata must fall back to weight 1.0 (never corrupt or crash the average).
    import numpy as np

    from nvflare.apis.dxo import MetaKey
    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    meta = {} if bad_steps is None else {MetaKey.NUM_STEPS_CURRENT_ROUND: bad_steps}
    aggregator.accept_model(FLModel(params={"w": np.array([2.0])}, meta=meta))
    aggregator.accept_model(FLModel(params={"w": np.array([4.0])}, meta=meta))
    result = aggregator.aggregate_model()

    # Both weights coerced to 1.0 -> plain mean (2 + 4) / 2 = 3.0.
    assert result.params["w"][0] == pytest.approx(3.0)


def test_custom_aggregator_template_resets_between_rounds():
    import numpy as np

    from nvflare.app_common.abstract.fl_model import FLModel

    module = _load_module(SHARED_TEMPLATES / "aggregator.py")
    aggregator = module.WeightedAggregator()

    aggregator.accept_model(FLModel(params={"w": np.array([1.0])}))
    aggregator.aggregate_model()

    # aggregate_model resets stats; a second aggregate with no accepts must fail.
    with pytest.raises(RuntimeError):
        aggregator.aggregate_model()


def test_lightning_eval_template_delivers_validation_metric_to_server():
    torch = pytest.importorskip("torch")
    pl = pytest.importorskip("pytorch_lightning")
    from torch.utils.data import DataLoader, TensorDataset

    module = _load_module(LIGHTNING_TEMPLATES / "lightning_client.py")

    class ToyLightning(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.layer(x)

        def validation_step(self, batch, batch_idx):
            features, labels = batch
            loss = torch.nn.functional.cross_entropy(self(features), labels)
            self.log("val_loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    loader = DataLoader(TensorDataset(torch.randn(6, 4), torch.randint(0, 2, (6,))), batch_size=3)
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False, devices=1)

    model = ToyLightning()
    metrics = module.validate_global_model(
        trainer,
        model,
        dataloaders=loader,
        make_higher_is_better=("val_loss",),
    )

    assert "val_loss" in metrics
    assert metrics["neg_val_loss"] == pytest.approx(-metrics["val_loss"])
    from nvflare.app_common.abstract.fl_model import FLModel, MetaKey
    from nvflare.app_common.utils.fl_model_utils import FLModelUtils

    assert model.__fl_meta__[MetaKey.INITIAL_METRICS] == metrics
    outgoing_model = FLModel(params=model.state_dict(), meta=model.__fl_meta__)
    server_model = FLModelUtils.from_shareable(FLModelUtils.to_shareable(outgoing_model))
    assert server_model.metrics == metrics


def test_lightning_template_eval_only_mode_skips_training():
    # FedEval / evaluation-only: main(evaluate_only=True) must validate but never
    # call trainer.fit, so a converted eval-only job does not train.
    module = _load_module(LIGHTNING_TEMPLATES / "lightning_client.py")

    calls = []

    class _FakeTrainer:
        callback_metrics = {"val_loss": 0.1}

        def validate(self, *a, **k):
            calls.append("validate")
            return [{"val_loss": 0.1}]

        def fit(self, *a, **k):
            calls.append("fit")

    fake = _FakeTrainer()

    class _FakeModel:
        pass

    import types

    fake_flare = types.SimpleNamespace(
        patch=lambda trainer: None,
        receive=lambda: None,
        _running=[True, False],
        is_running=lambda: fake_flare._running.pop(0) if fake_flare._running else False,
    )
    module.flare = fake_flare  # patch the module-level flare handle

    try:
        module.main(model=_FakeModel(), datamodule=object(), trainer_factory=lambda: fake, evaluate_only=True)
    finally:
        pass

    assert "validate" in calls
    assert "fit" not in calls


class _DummyModel:
    training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def __call__(self, *_args, **_kwargs):  # pragma: no cover - never reached on empty loader
        raise AssertionError("model should not be called when the loader is empty")


def test_lightning_negated_metric_helper_does_not_mutate_and_is_threaded_through_main():
    """The negation helper must be copy-safe, and main() must actually pass the keys.

    ``main`` is the round loop a generated ``client.py`` copies verbatim. If it does
    not forward ``make_higher_is_better`` to ``validate_global_model``, a lower-is-better
    conversion silently never delivers the negated key the recipe selects on.
    """
    import inspect as _inspect

    module = _load_module(LIGHTNING_TEMPLATES / "lightning_client.py")

    source = {"val_loss": 0.25, "val_acc": 0.9}
    negated = module.add_higher_is_better_metrics(source, ("val_loss",))

    assert negated == {"val_loss": 0.25, "val_acc": 0.9, "neg_val_loss": -0.25}
    assert source == {"val_loss": 0.25, "val_acc": 0.9}, "helper must not mutate its input"

    with pytest.raises(RuntimeError, match="not in the validation results"):
        module.add_higher_is_better_metrics({"val_acc": 1.0}, ("val_loss",))
    with pytest.raises(RuntimeError, match="already exists"):
        module.add_higher_is_better_metrics({"val_loss": 1.0, "neg_val_loss": 0.0}, ("val_loss",))

    assert "make_higher_is_better" in _inspect.signature(module.main).parameters
    main_source = _inspect.getsource(module.main)
    assert "make_higher_is_better=make_higher_is_better" in main_source
