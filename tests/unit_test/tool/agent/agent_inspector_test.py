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

import os
from pathlib import Path

import pytest

from nvflare.tool.agent.inspector import inspect_source


def write_project(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


@pytest.mark.parametrize(
    ("source", "skill", "framework"),
    [
        (
            "from transformers import Trainer\ntrainer = Trainer()\ntrainer.train()\n",
            "nvflare-convert-huggingface",
            "huggingface",
        ),
        (
            "import lightning.pytorch as pl\ntrainer = pl.Trainer()\ntrainer.fit(model)\n",
            "nvflare-convert-lightning",
            "lightning",
        ),
        (
            "import torch\noptimizer = torch.optim.Adam(model.parameters())\noptimizer.step()\n",
            "nvflare-convert-pytorch",
            "pytorch",
        ),
    ],
)
def test_direct_converter_owner(tmp_path, source, skill, framework):
    write_project(tmp_path, {"train.py": source})

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "clear"
    assert result["ownership"]["framework"] == framework
    assert result["routing"] == {"recommended_skill": skill, "reason": "clear_owner"}


@pytest.mark.parametrize("method", ["fit", "validate", "test"])
def test_compact_lightning_owners(tmp_path, method):
    write_project(tmp_path, {"eval.py": f"import lightning as L\nL.Trainer().{method}(model)\n"})

    result = inspect_source(tmp_path)

    assert result["ownership"]["framework"] == "lightning"
    assert result["routing"]["recommended_skill"] == "nvflare-convert-lightning"


@pytest.mark.parametrize("optimizer", ["Adagrad", "Adam", "AdamW", "RMSprop", "SGD"])
def test_closed_optimizer_table(tmp_path, optimizer):
    write_project(
        tmp_path,
        {"train.py": f"from torch.optim import {optimizer} as Opt\noptimizer = Opt(params)\noptimizer.step()\n"},
    )

    assert inspect_source(tmp_path)["ownership"]["framework"] == "pytorch"


@pytest.mark.parametrize(
    "source",
    [
        "import torch\ndef train(optimizer: torch.optim.Optimizer):\n    optimizer.step()\n",
        "from torch.optim import Optimizer as OptimizerBase\n"
        "def train(optimizer: OptimizerBase):\n    optimizer.step()\n",
    ],
)
def test_typed_optimizer_parameter_is_pytorch_owner(tmp_path, source):
    write_project(tmp_path, {"train.py": source})

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "clear"
    assert result["ownership"]["framework"] == "pytorch"
    assert result["routing"] == {"recommended_skill": "nvflare-convert-pytorch", "reason": "clear_owner"}


@pytest.mark.parametrize("step", ["scaler.step(optimizer)", "scaler.step(optimizer=optimizer)"])
def test_closed_grad_scaler_pair(tmp_path, step):
    write_project(
        tmp_path,
        {
            "train.py": "from torch.optim import SGD\nfrom torch.amp import GradScaler\n"
            "optimizer = SGD(params)\nscaler = GradScaler()\n"
            f"{step}\n"
        },
    )

    assert inspect_source(tmp_path)["ownership"]["framework"] == "pytorch"


def test_supporting_only_routes_orient(tmp_path):
    write_project(tmp_path, {"eval.py": "from transformers import Trainer\nt = Trainer()\nt.evaluate()\n"})

    result = inspect_source(tmp_path)

    assert result["ownership"]["reason"] == "supporting_only"


def test_pytorch_backward_is_supporting_only(tmp_path):
    write_project(tmp_path, {"train.py": "import torch\nloss.backward()\n"})

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["ownership"]["reason"] == "supporting_only"
    assert result["ownership"]["candidate_frameworks"] == ["pytorch"]
    assert result["routing"]["recommended_skill"] == "nvflare-orient"
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


def test_model_and_import_only_do_not_select_converter(tmp_path):
    write_project(
        tmp_path,
        {"model.py": "import torch\nclass Net(torch.nn.Module):\n    pass\nmodel = Net()\n"},
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "none"
    assert result["routing"] == {"recommended_skill": None, "reason": "no_route"}


def test_same_framework_owners_route_converter_but_require_narrowing(tmp_path):
    source = "from transformers import Trainer\nt = Trainer()\nt.train()\n"
    write_project(tmp_path, {"a.py": source, "b.py": source})

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "clear"
    assert result["ownership"]["owner_file"] is None
    assert result["ownership"]["candidate_files"] == ["a.py", "b.py"]
    assert result["routing"]["recommended_skill"] == "nvflare-convert-huggingface"


def test_cross_framework_owners_route_orient(tmp_path):
    write_project(
        tmp_path,
        {
            "hf.py": "from transformers import Trainer\nt = Trainer()\nt.train()\n",
            "pt.py": "from torch.optim import Adam\no = Adam(params)\no.step()\n",
        },
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "conflicting"
    assert result["routing"]["reason"] == "conflicting_owner"


@pytest.mark.parametrize(
    "integration",
    [
        "import nvflare.client.hf as flare\nflare.patch(t)\n",
        "import nvflare.client as flare\nflare.receive()\nflare.send(result)\n",
        "from nvflare.client.api import FLModel\nmodel = FLModel()\n",
        "from nvflare.client import FLModel\nclass Result(FLModel):\n    pass\n",
    ],
)
def test_direct_client_api_suppresses_conversion(tmp_path, integration):
    write_project(
        tmp_path,
        {"train.py": "from transformers import Trainer\nt = Trainer()\nt.train()\n" + integration},
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "converted"
    assert result["routing"] == {"recommended_skill": None, "reason": "already_integrated"}


@pytest.mark.parametrize(
    "client_import, class_body",
    [
        ("import nvflare.client.hf as flare", "result = flare.patch(trainer)"),
        ("from nvflare.client import FLModel", "result = FLModel()"),
        ("import nvflare.client as flare", "received = flare.receive()\n    flare.send(received)"),
    ],
)
def test_direct_client_api_in_class_body_suppresses_conversion(tmp_path, client_import, class_body):
    write_project(
        tmp_path,
        {
            "train.py": "from torch.optim import SGD\n"
            f"{client_import}\n"
            "o = SGD(params)\no.step()\n"
            f"class Converted:\n    {class_body}\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "converted"
    assert result["routing"] == {"recommended_skill": None, "reason": "already_integrated"}


@pytest.mark.parametrize(
    "class_body",
    [
        "if ready:\n        flare = local\n        flare.patch(trainer)",
        "callback = lambda: flare.patch(trainer)",
        "results = [flare.patch(trainer) for trainer in trainers]",
    ],
)
def test_unsupported_class_body_client_calls_are_possible(tmp_path, class_body):
    write_project(
        tmp_path,
        {
            "train.py": "from torch.optim import SGD\n"
            "import nvflare.client.hf as flare\n"
            "o = SGD(params)\no.step()\n"
            f"class Converted:\n    {class_body}\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "possible"
    assert result["routing"] == {"recommended_skill": "nvflare-orient", "reason": "possible_integration"}


def test_conditional_class_body_client_import_does_not_become_credible(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "class Client:\n"
            "    if enabled:\n"
            "        import nvflare.client as flare\n"
            "    flare.patch(trainer)\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "possible"
    assert result["routing"] == {"recommended_skill": "nvflare-orient", "reason": "possible_integration"}


def test_conditional_function_client_import_does_not_become_credible(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from torch.optim import SGD\n"
            "def run():\n"
            "    optimizer = SGD(params)\n"
            "    optimizer.step()\n"
            "    if enabled:\n"
            "        import nvflare.client as flare\n"
            "    flare.patch(trainer)\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "possible"
    assert result["routing"] == {"recommended_skill": "nvflare-orient", "reason": "possible_integration"}


@pytest.mark.parametrize(
    "conditional_body",
    [
        "import nvflare.client as flare\n        flare.patch(trainer)",
        "from nvflare.client import FLModel\n        FLModel()",
        "from nvflare.client import FLModel\n        class FLModel(FLModel):\n            pass",
    ],
)
def test_client_import_and_call_in_same_conditional_branch_are_possible(tmp_path, conditional_body):
    write_project(
        tmp_path,
        {
            "train.py": "from torch.optim import SGD\n"
            "optimizer = SGD(params)\noptimizer.step()\n"
            f"if enabled:\n        {conditional_body}\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "possible"
    assert result["routing"] == {"recommended_skill": "nvflare-orient", "reason": "possible_integration"}


@pytest.mark.parametrize(
    ("client_import", "conditional_body"),
    [
        ("import nvflare.client as flare", "flare.patch(trainer)"),
        ("from nvflare.client import FLModel", "class Result(FLModel):\n            pass"),
    ],
)
def test_unconditional_client_import_used_in_conditional_branch_remains_credible(
    tmp_path, client_import, conditional_body
):
    write_project(
        tmp_path,
        {
            "train.py": "from torch.optim import SGD\n"
            f"{client_import}\n"
            "optimizer = SGD(params)\noptimizer.step()\n"
            f"if enabled:\n        {conditional_body}\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "converted"
    assert result["routing"] == {"recommended_skill": None, "reason": "already_integrated"}


def test_conditional_class_body_flmodel_import_does_not_become_credible(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from torch.optim import SGD\n"
            "optimizer = SGD(params)\noptimizer.step()\n"
            "class Client:\n"
            "    if enabled:\n"
            "        from nvflare.client import FLModel\n"
            "    class Result(FLModel):\n"
            "        pass\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "possible"
    assert result["routing"] == {"recommended_skill": "nvflare-orient", "reason": "possible_integration"}


def test_direct_flmodel_subclass_nested_in_class_body_suppresses_conversion(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from torch.optim import SGD\n"
            "from nvflare.client import FLModel\n"
            "o = SGD(params)\no.step()\n"
            "class Outer:\n"
            "    class Result(FLModel):\n"
            "        pass\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "converted"
    assert result["routing"] == {"recommended_skill": None, "reason": "already_integrated"}


def test_partial_client_api_fails_closed(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from torch.optim import Adam\nimport nvflare.client as flare\n"
            "o = Adam(params)\no.step()\nflare.receive()\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "possible"
    assert result["routing"]["reason"] == "possible_integration"


@pytest.mark.parametrize(
    "client_import, calls",
    [
        ("from nvflare.client import patch", "patch(None)"),
        ("import nvflare.client as flare", "flare.patch(None)"),
        ("from nvflare.client import receive, send", "receive()\n        send(result)"),
    ],
)
def test_nested_function_client_alias_is_never_lost(tmp_path, client_import, calls):
    write_project(
        tmp_path,
        {
            "train.py": f"{client_import}\n"
            "from transformers import Trainer\n"
            "trainer = Trainer()\ntrainer.train()\n"
            "def outer():\n"
            "    def inner():\n"
            f"        {calls}\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "possible"
    assert result["routing"]["reason"] == "possible_integration"


def test_split_receive_send_is_not_joined_into_credible_integration(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from torch.optim import SGD\no = SGD(params)\no.step()\n",
            "receive.py": "import nvflare.client as flare\nflare.receive()\n",
            "send.py": "import nvflare.client as flare\nflare.send(result)\n",
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] != "converted"
    assert result["routing"]["recommended_skill"] == "nvflare-convert-pytorch"


def test_owner_import_reaches_package_initializer_client_api(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from torch.optim import SGD\nimport package.helper\no = SGD(params)\no.step()\n",
            "package/__init__.py": "import nvflare.client as flare\nflare.receive()\nflare.send(result)\n",
            "package/helper.py": "VALUE = 1\n",
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "converted"
    assert result["routing"]["reason"] == "already_integrated"


@pytest.mark.parametrize(
    ("entry", "active_helper", "stale_helper"),
    [
        ("train.py", "helper.py", "src/helper.py"),
        ("src/train.py", "src/helper.py", "helper.py"),
    ],
)
def test_importer_packaging_root_prevents_stale_client_api_authority(tmp_path, entry, active_helper, stale_helper):
    write_project(
        tmp_path,
        {
            entry: "import helper\nfrom torch.optim import SGD\no = SGD(params)\no.step()\n",
            active_helper: "VALUE = 1\n",
            stale_helper: "import nvflare.client as flare\nflare.patch(trainer)\n",
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "none"
    assert result["routing"] == {"recommended_skill": "nvflare-convert-pytorch", "reason": "clear_owner"}


@pytest.mark.parametrize(
    ("entry", "active_helper", "stale_helper"),
    [
        ("job.py", "helper.py", "src/helper.py"),
        ("src/job.py", "src/helper.py", "helper.py"),
    ],
)
def test_importer_packaging_root_prevents_stale_source_job_authority(tmp_path, entry, active_helper, stale_helper):
    write_project(
        tmp_path,
        {
            entry: "import helper\n",
            active_helper: "VALUE = 1\n",
            stale_helper: "import nvflare\n",
        },
    )

    result = inspect_source(tmp_path)

    assert result["routing"] == {"recommended_skill": None, "reason": "no_route"}


def test_pr4955_nested_fixture_does_not_suppress_root_owner(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from torch.optim import Adam\no = Adam(params)\no.step()\n",
            "tests/fixture/job.py": "from nvflare.job_config.api import FedJob\njob = FedJob(name='old')\n",
        },
    )

    result = inspect_source(tmp_path)

    assert result["routing"]["recommended_skill"] == "nvflare-convert-pytorch"


def test_pr4955_reachable_delegated_source_job_suppresses_conversion(tmp_path):
    write_project(
        tmp_path,
        {
            "job.py": "from helper import build_job\njob = build_job()\n",
            "helper.py": "from nvflare.job_config.api import FedJob\ndef build_job():\n    return FedJob(name='job')\n",
        },
    )

    assert inspect_source(tmp_path)["routing"] == {"recommended_skill": None, "reason": "existing_job"}


def test_pr4955_unreachable_helper_does_not_corroborate_job(tmp_path):
    write_project(
        tmp_path,
        {
            "job.py": "def build_job():\n    return object()\n",
            "unused.py": "from nvflare.job_config.api import FedJob\n",
        },
    )

    assert inspect_source(tmp_path)["routing"] == {"recommended_skill": None, "reason": "no_route"}


def test_root_launcher_to_nested_job_is_authoritative(tmp_path):
    write_project(
        tmp_path,
        {
            "main.py": "from jobs.active import job\n",
            "jobs/active/job.py": "from nvflare.job_config.api import FedJob\njob = FedJob(name='active')\n",
            "jobs/active/__init__.py": "from .job import job\n",
        },
    )

    assert inspect_source(tmp_path)["routing"]["reason"] == "existing_job"


def test_job_import_reaches_package_initializer_nvflare_evidence(tmp_path):
    write_project(
        tmp_path,
        {
            "job.py": "import package.helper\njob = object()\n",
            "package/__init__.py": "import nvflare\n",
            "package/helper.py": "VALUE = 1\n",
        },
    )

    assert inspect_source(tmp_path)["routing"]["reason"] == "existing_job"


def test_root_launcher_authorizes_reached_secondary_job(tmp_path):
    write_project(
        tmp_path,
        {
            "main.py": "from tests.fixture import job\n",
            "tests/__init__.py": "",
            "tests/fixture/__init__.py": "from .job import job\n",
            "tests/fixture/job.py": "from nvflare.job_config.api import FedJob\njob = FedJob(name='fixture')\n",
        },
    )

    assert inspect_source(tmp_path)["routing"]["reason"] == "existing_job"


def test_connected_root_launcher_chain_is_one_authority(tmp_path):
    write_project(
        tmp_path,
        {
            "main.py": "import launch\n",
            "launch.py": "from tests.fixture import job\n",
            "tests/__init__.py": "",
            "tests/fixture/__init__.py": "from .job import job\n",
            "tests/fixture/job.py": "import nvflare\njob = object()\n",
        },
    )

    assert inspect_source(tmp_path)["routing"]["reason"] == "existing_job"


def test_independent_root_launchers_to_secondary_job_fail_closed(tmp_path):
    write_project(
        tmp_path,
        {
            "one.py": "from tests.fixture import job\n",
            "two.py": "from tests.fixture import job\n",
            "tests/__init__.py": "",
            "tests/fixture/__init__.py": "from .job import job\n",
            "tests/fixture/job.py": "import nvflare\njob = object()\n",
        },
    )

    result = inspect_source(tmp_path)

    assert result["routing"]["reason"] == "possible_existing_job"


def test_root_launcher_selects_one_of_multiple_nested_jobs(tmp_path):
    write_project(
        tmp_path,
        {
            "main.py": "from jobs.a import job\n",
            "jobs/a/job.py": "import nvflare\njob = object()\n",
            "jobs/a/__init__.py": "from .job import job\n",
            "jobs/b/job.py": "import nvflare\njob = object()\n",
        },
    )

    assert inspect_source(tmp_path)["routing"]["reason"] == "existing_job"


@pytest.mark.parametrize(
    "job_source",
    [
        "class Builder:\n    from nvflare.job_config.api import FedJob\n",
        "class Builder:\n    from helper import build_job\n",
    ],
)
def test_class_body_imports_corroborate_source_job(tmp_path, job_source):
    write_project(tmp_path, {"job.py": job_source, "helper.py": "import nvflare\n"})

    assert inspect_source(tmp_path)["routing"]["reason"] == "existing_job"


def test_secondary_subtree_is_positive_when_inspected_directly(tmp_path):
    fixture = tmp_path / "tests" / "fixture"
    write_project(
        fixture,
        {"job.py": "from nvflare.job_config.api import FedJob\njob = FedJob(name='fixture')\n"},
    )

    assert inspect_source(fixture)["routing"]["reason"] == "existing_job"


@pytest.mark.parametrize(
    "files",
    [
        {
            "train.py": "from torch.optim import SGD\no = SGD(params)\no.step()\n",
            "tests/fixture/job.py": "from train import o\nfrom nvflare.job_config.api import FedJob\n",
        },
        {
            "common.py": "VALUE = 1\n",
            "app/train.py": "from common import VALUE\nfrom torch.optim import SGD\no = SGD(params)\no.step()\n",
            "archive/job.py": "from common import VALUE\nfrom nvflare.job_config.api import FedJob\n",
        },
    ],
)
def test_pr4955_reverse_and_shared_helper_do_not_grant_job_authority(tmp_path, files):
    write_project(tmp_path, files)

    assert inspect_source(tmp_path)["routing"]["recommended_skill"] == "nvflare-convert-pytorch"


def test_pr4955_unique_contained_job_is_authoritative(tmp_path):
    write_project(tmp_path, {"jobs/fedavg/job.py": "from nvflare.job_config.api import FedJob\n"})

    assert inspect_source(tmp_path)["routing"]["reason"] == "existing_job"


def test_pr4955_nested_converted_job_keeps_authority_beside_model_only_root(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "import torch\ndef train():\n    return torch.nn.Linear(1, 1)\n",
            "jobs/fedavg/job.py": "from nvflare.app_common.workflows.fedavg import FedAvg\n"
            "from nvflare.job_config.api import FedJob\n"
            "job = FedJob()\ncontroller = FedAvg()\njob.export()\n",
        },
    )

    assert inspect_source(tmp_path)["routing"] == {"recommended_skill": None, "reason": "existing_job"}


def test_independent_nested_job_beside_active_owner_fails_closed(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from torch.optim import SGD\noptimizer = SGD(params)\noptimizer.step()\n",
            "jobs/fedavg/job.py": "from nvflare.job_config.api import FedJob\njob = FedJob()\n",
        },
    )

    assert inspect_source(tmp_path)["routing"] == {
        "recommended_skill": "nvflare-orient",
        "reason": "possible_existing_job",
    }


@pytest.mark.parametrize(
    "model_source",
    [
        "import torch\nclass Net(torch.nn.Module):\n    pass\n",
        "from torch.optim import SGD\nclass Net:\n    pass\noptimizer = SGD(params)\noptimizer.step()\n",
    ],
)
def test_only_project_under_secondary_tree_fails_closed(tmp_path, model_source):
    write_project(
        tmp_path,
        {
            "tests/integration/model.py": model_source,
            "tests/integration/job.py": "from model import Net\n"
            "from nvflare.job_config.api import FedJob\n"
            "job = FedJob()\nnet = Net()\n",
        },
    )

    assert inspect_source(tmp_path)["routing"] == {
        "recommended_skill": "nvflare-orient",
        "reason": "possible_existing_job",
    }


def test_pr4955_multiple_independent_jobs_fail_closed(tmp_path):
    write_project(
        tmp_path,
        {
            "jobs/a/job.py": "from nvflare.job_config.api import FedJob\n",
            "jobs/b/job.py": "from nvflare.job_config.api import FedJob\n",
        },
    )

    assert inspect_source(tmp_path)["routing"] == {
        "recommended_skill": "nvflare-orient",
        "reason": "possible_existing_job",
    }


@pytest.mark.parametrize(
    "source",
    [
        "from transformers import Trainer\ndef make():\n    return Trainer()\nt = make()\nt.train()\n",
        "from transformers import Trainer\nt = Trainer()\ndef run():\n    t.train()\n",
    ],
)
def test_factory_and_cross_scope_owner_attempts_are_unresolved(tmp_path, source):
    write_project(tmp_path, {"train.py": source})

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["ownership"]["reason"] == "unsupported_indirection"
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


def test_incomplete_scan_fails_closed_before_converter(tmp_path):
    write_project(
        tmp_path,
        {
            "a.py": "from transformers import Trainer\nt = Trainer()\nt.train()\n",
            "b.py": "print('not scanned')\n",
        },
    )

    result = inspect_source(tmp_path, max_files=1)

    assert result["scan"]["complete"] is False
    assert result["routing"]["recommended_skill"] == "nvflare-orient"
    assert result["routing"]["reason"] == "possible_existing_job"


def test_exact_exported_job_suppresses_conversion(tmp_path):
    write_project(tmp_path, {"meta.json": "{}", "app/config/config_fed_server.json": "{}"})

    assert inspect_source(tmp_path)["routing"] == {"recommended_skill": None, "reason": "existing_job"}


def test_incomplete_exported_job_markers_fail_closed(tmp_path):
    write_project(tmp_path, {"meta.json": "{}"})

    assert inspect_source(tmp_path)["routing"]["reason"] == "possible_existing_job"


def test_job_filename_without_nvflare_import_is_not_a_job(tmp_path):
    write_project(tmp_path, {"job.py": "def submit():\n    return 'slurm'\n"})

    assert inspect_source(tmp_path)["routing"]["reason"] == "no_route"


def test_relative_local_nvflare_module_does_not_corroborate_source_job(tmp_path):
    write_project(
        tmp_path,
        {
            "job.py": "from .nvflare import helper\njob = object()\n",
            "nvflare.py": "helper = object()\n",
            "train.py": "from torch.optim import SGD\no = SGD(params)\no.step()\n",
        },
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["framework"] == "pytorch"
    assert result["routing"] == {"recommended_skill": "nvflare-convert-pytorch", "reason": "clear_owner"}


def test_absolute_local_nvflare_package_is_not_client_api_or_job_evidence(tmp_path):
    write_project(
        tmp_path,
        {
            "job.py": "import nvflare\n",
            "nvflare/__init__.py": "VALUE = 1\n",
            "nvflare/client.py": "def patch(value):\n    return value\n",
            "train.py": "import nvflare.client as flare\n"
            "from torch.optim import SGD\n"
            "flare.patch(trainer)\n"
            "o = SGD(params)\no.step()\n",
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "none"
    assert result["routing"] == {"recommended_skill": "nvflare-convert-pytorch", "reason": "clear_owner"}


def test_absolute_local_nvflare_package_root_shadows_missing_submodule(tmp_path):
    write_project(
        tmp_path,
        {
            "nvflare/__init__.py": "VALUE = 1\n",
            "train.py": "import nvflare.client as flare\n"
            "from torch.optim import SGD\n"
            "flare.patch(trainer)\n"
            "o = SGD(params)\no.step()\n",
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "none"
    assert result["routing"] == {"recommended_skill": "nvflare-convert-pytorch", "reason": "clear_owner"}


@pytest.mark.parametrize(
    ("training_path", "stale_path"),
    [("train.py", "src/nvflare.py"), ("src/train.py", "nvflare.py")],
)
def test_nvflare_module_in_other_packaging_root_does_not_shadow_client_api(tmp_path, training_path, stale_path):
    write_project(
        tmp_path,
        {
            stale_path: "VALUE = 1\n",
            training_path: "import nvflare.client as flare\n"
            "from torch.optim import SGD\n"
            "flare.patch(trainer)\n"
            "o = SGD(params)\no.step()\n",
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "converted"
    assert result["routing"] == {"recommended_skill": None, "reason": "already_integrated"}


def test_file_target_sees_sibling_local_nvflare_package_root(tmp_path):
    write_project(
        tmp_path,
        {
            "nvflare/__init__.py": "VALUE = 1\n",
            "train.py": "import nvflare.client as flare\n"
            "from torch.optim import SGD\n"
            "flare.patch(trainer)\n"
            "o = SGD(params)\no.step()\n",
        },
    )

    result = inspect_source(tmp_path / "train.py")

    assert result["integration"]["state"] == "none"
    assert result["routing"] == {"recommended_skill": "nvflare-convert-pytorch", "reason": "clear_owner"}


def test_symlinked_local_nvflare_package_root_shadows_installed_package(tmp_path):
    write_project(
        tmp_path,
        {
            "real_nvflare/__init__.py": "VALUE = 1\n",
            "train.py": "import nvflare.client as flare\n"
            "from torch.optim import SGD\n"
            "flare.patch(trainer)\n"
            "o = SGD(params)\no.step()\n",
        },
    )
    try:
        (tmp_path / "nvflare").symlink_to(tmp_path / "real_nvflare", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "none"
    assert result["routing"] == {"recommended_skill": "nvflare-convert-pytorch", "reason": "clear_owner"}


def test_unparseable_local_nvflare_package_root_fails_closed(tmp_path):
    write_project(
        tmp_path,
        {
            "nvflare/__init__.py": "def broken(:\n",
            "train.py": "import nvflare.client as flare\n"
            "from torch.optim import SGD\n"
            "flare.patch(trainer)\n"
            "o = SGD(params)\no.step()\n",
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] != "converted"
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


def test_non_file_nvflare_module_candidate_does_not_shadow_installed_package(tmp_path):
    write_project(
        tmp_path,
        {
            "nvflare.py/placeholder.txt": "not a Python module\n",
            "train.py": "import nvflare.client as flare\n"
            "from torch.optim import SGD\n"
            "flare.patch(trainer)\n"
            "o = SGD(params)\no.step()\n",
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "converted"
    assert result["routing"] == {"recommended_skill": None, "reason": "already_integrated"}


@pytest.mark.parametrize(
    "source",
    [
        "from .transformers import Trainer\nt = Trainer()\nt.train()\n",
        "from .lightning import Trainer\nt = Trainer()\nt.fit(model)\n",
        "from .torch.optim import SGD\no = SGD(params)\no.step()\n",
    ],
)
def test_relative_local_framework_names_do_not_select_converter(tmp_path, source):
    write_project(tmp_path, {"train.py": source})

    assert inspect_source(tmp_path)["routing"]["reason"] == "no_route"


def test_owner_scan_parses_each_python_file_once(tmp_path, monkeypatch):
    write_project(
        tmp_path,
        {
            "train.py": "from torch.optim import SGD\no = SGD(params)\no.step()\n",
            "helper.py": "VALUE = 1\n",
        },
    )
    import nvflare.tool.agent.inspection.files as source_files

    original = source_files.ast.parse
    counts = []

    def counted(*args, **kwargs):
        counts.append(kwargs.get("filename", args[1] if len(args) > 1 else None))
        return original(*args, **kwargs)

    monkeypatch.setattr(source_files.ast, "parse", counted)

    inspect_source(tmp_path)

    assert sorted(counts) == ["helper.py", "train.py"]


@pytest.mark.parametrize(
    "source",
    [
        "from transformers import Trainer\nt = Trainer()\nresult = t.train()\n",
        "from transformers import Trainer\ndef run():\n    t = Trainer()\n    return t.train()\n",
        "from transformers import Trainer\nasync def run():\n    t = Trainer()\n    await t.train()\n",
        "from transformers import Trainer\nt = Trainer()\nif t.train():\n    pass\n",
        "from transformers import Trainer\nt = Trainer()\nconsume(t.train())\n",
    ],
)
def test_direct_owner_calls_are_found_in_expression_contexts(tmp_path, source):
    write_project(tmp_path, {"train.py": source})

    assert inspect_source(tmp_path)["ownership"]["framework"] == "huggingface"


@pytest.mark.parametrize("statement", ["assert trainer.train()", "raise trainer.train()"])
def test_direct_owner_calls_in_assert_and_raise_are_found(tmp_path, statement):
    write_project(
        tmp_path,
        {"train.py": f"from transformers import Trainer\ntrainer = Trainer()\n{statement}\n"},
    )

    assert inspect_source(tmp_path)["ownership"]["framework"] == "huggingface"


def test_returned_flmodel_suppresses_clear_owner(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from torch.optim import SGD\nfrom nvflare.client.api import FLModel\n"
            "o = SGD(params)\no.step()\n"
            "def result():\n    return FLModel()\n"
        },
    )

    assert inspect_source(tmp_path)["routing"]["reason"] == "already_integrated"


def test_returned_app_common_flmodel_suppresses_clear_owner(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from torch.optim import SGD\n"
            "from nvflare.app_common.abstract.fl_model import FLModel\n"
            "o = SGD(params)\no.step()\n"
            "def result():\n    return FLModel()\n"
        },
    )

    assert inspect_source(tmp_path)["routing"]["reason"] == "already_integrated"


def test_local_function_rebinds_imported_trainer(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from transformers import Trainer\n"
            "def Trainer():\n    return object()\n"
            "t = Trainer()\nt.train()\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "none"
    assert result["routing"]["recommended_skill"] is None


def test_later_function_local_binding_shadows_module_import_for_entire_scope(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from transformers import Trainer\n"
            "def run():\n"
            "    trainer = Trainer()\n"
            "    Trainer = local\n"
            "    trainer.train()\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "none"
    assert result["routing"] == {"recommended_skill": None, "reason": "no_route"}


def test_later_module_binding_shadows_global_constructor_used_by_function(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from transformers import Trainer\n"
            "def run():\n"
            "    trainer = Trainer()\n"
            "    trainer.train()\n"
            "Trainer = local\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "none"
    assert result["routing"] == {"recommended_skill": None, "reason": "no_route"}


@pytest.mark.parametrize(
    "source",
    [
        "from transformers import Trainer\n"
        "def run():\n    trainer = Trainer()\n    trainer.train()\n"
        "import transformers as Trainer\n",
        "from lightning import Trainer\n"
        "def run():\n    trainer = Trainer()\n    trainer.fit(model)\n"
        "import lightning as Trainer\n",
        "from torch.optim import SGD\n"
        "def run():\n    optimizer = SGD(params)\n    optimizer.step()\n"
        "import torch as SGD\n",
    ],
)
def test_framework_module_rebinding_does_not_validate_inherited_constructor(tmp_path, source):
    write_project(tmp_path, {"train.py": source})

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "none"
    assert result["routing"] == {"recommended_skill": None, "reason": "no_route"}


@pytest.mark.parametrize(
    ("source", "framework", "skill"),
    [
        (
            "import transformers\n" "def run():\n    trainer = transformers.Trainer()\n    trainer.train()\n",
            "huggingface",
            "nvflare-convert-huggingface",
        ),
        (
            "import lightning\n" "def run():\n    trainer = lightning.Trainer()\n    trainer.fit(model)\n",
            "lightning",
            "nvflare-convert-lightning",
        ),
        (
            "import torch\n" "def run():\n    optimizer = torch.optim.SGD(params)\n    optimizer.step()\n",
            "pytorch",
            "nvflare-convert-pytorch",
        ),
    ],
)
def test_qualified_inherited_constructor_keeps_exact_dependency(tmp_path, source, framework, skill):
    write_project(tmp_path, {"train.py": source})

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "clear"
    assert result["ownership"]["framework"] == framework
    assert result["routing"] == {"recommended_skill": skill, "reason": "clear_owner"}


@pytest.mark.parametrize(
    "replacement",
    [
        "Trainer = local\nfrom transformers import Trainer\n",
        "class Trainer(Trainer):\n    pass\n",
    ],
)
def test_same_framework_module_replacement_preserves_function_owner(tmp_path, replacement):
    write_project(
        tmp_path,
        {
            "train.py": "from transformers import Trainer\n"
            f"{replacement}"
            "def run():\n"
            "    trainer = Trainer()\n"
            "    trainer.train()\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "clear"
    assert result["ownership"]["framework"] == "huggingface"
    assert result["routing"] == {"recommended_skill": "nvflare-convert-huggingface", "reason": "clear_owner"}


@pytest.mark.parametrize(
    ("source", "framework", "skill"),
    [
        (
            "from transformers import Trainer, Seq2SeqTrainer\n"
            "def run():\n"
            "    trainer = Trainer()\n"
            "    trainer.train()\n"
            "    Seq2SeqTrainer = object\n",
            "huggingface",
            "nvflare-convert-huggingface",
        ),
        (
            "import lightning as L\n"
            "from lightning import Trainer\n"
            "def run():\n"
            "    trainer = L.Trainer()\n"
            "    trainer.fit(model)\n"
            "    Trainer = object\n",
            "lightning",
            "nvflare-convert-lightning",
        ),
    ],
)
def test_unrelated_same_framework_shadow_does_not_erase_owner(tmp_path, source, framework, skill):
    write_project(tmp_path, {"train.py": source})

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "clear"
    assert result["ownership"]["framework"] == framework
    assert result["routing"] == {"recommended_skill": skill, "reason": "clear_owner"}


@pytest.mark.parametrize(
    "source",
    [
        "from torch.optim import SGD\no = SGD(params)\nif ready:\n    o.step()\n",
        "from torch.optim import SGD\no = SGD(params)\nfor batch in data:\n    o.step()\n",
        "from transformers import Trainer\nt = Trainer()\ntry:\n    t.train()\nexcept Exception:\n    pass\n",
    ],
)
def test_control_flow_blocks_preserve_same_scope_instances(tmp_path, source):
    write_project(tmp_path, {"train.py": source})

    assert inspect_source(tmp_path)["ownership"]["state"] == "clear"


def test_direct_file_job_target_keeps_source_marker(tmp_path):
    job = tmp_path / "job.py"
    job.write_text("from nvflare.job_config.api import FedJob\njob = FedJob(name='active')\n", encoding="utf-8")

    assert inspect_source(job)["routing"]["reason"] == "existing_job"


def test_multiple_independent_client_api_candidates_are_ambiguous(tmp_path):
    write_project(
        tmp_path,
        {
            "a.py": "import nvflare.client.hf as flare\nflare.patch(a)\n",
            "b.py": "import nvflare.client.lightning as flare\nflare.patch(b)\n",
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "possible"
    assert result["integration"]["complete"] is False
    assert result["integration"]["reason"] == "ambiguous_client_api"
    assert {item["file"] for item in result["integration"]["evidence"]} == {"a.py", "b.py"}
    assert result["routing"]["reason"] == "possible_integration"


def test_module_imports_are_not_propagated_beyond_direct_child_function(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from transformers import Trainer\n"
            "def outer():\n"
            "    def inner():\n"
            "        t = Trainer()\n"
            "        t.train()\n"
        },
    )

    assert inspect_source(tmp_path)["ownership"]["state"] == "none"


@pytest.mark.parametrize(
    "source",
    [
        "from transformers import TrainingArguments\nargs = TrainingArguments('out')\n"
        "trainer = build_trainer(args)\ntrainer.train()\n",
        "from transformers import Seq2SeqTrainingArguments\nargs = Seq2SeqTrainingArguments('out')\n"
        "trainer = build_trainer(args)\ntrainer.train()\n",
        "from trl import SFTConfig\nargs = SFTConfig(output_dir='out')\n"
        "trainer = build_trainer(args)\ntrainer.train()\n",
    ],
)
def test_hf_config_plus_unpaired_train_is_unresolved_not_clear(tmp_path, source):
    write_project(tmp_path, {"train.py": source})

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["ownership"]["candidate_frameworks"] == ["huggingface"]
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


def test_hf_config_alone_does_not_select_a_converter(tmp_path):
    write_project(
        tmp_path, {"config.py": "from transformers import TrainingArguments\nargs = TrainingArguments('out')\n"}
    )

    assert inspect_source(tmp_path)["routing"] == {"recommended_skill": None, "reason": "no_route"}


@pytest.mark.parametrize(
    "source",
    [
        "from transformers import TrainingArguments\ntrainer = build_trainer()\ntrainer.train()\n",
        "from transformers import TrainingArguments\nargs = TrainingArguments('out')\n"
        "trainer = build_trainer(args)\ntrainer.evaluate()\n",
    ],
)
def test_hf_config_context_does_not_broaden_beyond_constructed_train(tmp_path, source):
    write_project(tmp_path, {"train.py": source})

    assert inspect_source(tmp_path)["routing"] == {"recommended_skill": None, "reason": "no_route"}


@pytest.mark.parametrize(
    "source",
    [
        "from transformers import Trainer\n@decorate\ndef run():\n    t = Trainer()\n    t.train()\n",
        "from transformers import Trainer\ndef run():\n    t = Trainer()\n    t.train()\n    yield 1\n",
        "from transformers import Trainer\ndef run():\n    t = Trainer()\n    t.train()\n    x = (yield 1)\n",
        "from transformers import Trainer\ndef run():\n    t = Trainer()\n    t.train()\n    x = (yield from values)\n",
    ],
)
def test_decorated_and_generator_owner_forms_fail_closed(tmp_path, source):
    write_project(tmp_path, {"train.py": source})

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


@pytest.mark.parametrize(
    "source",
    [
        "from transformers import Trainer\nt = Trainer()\nrun = lambda: t.train()\n",
        "from transformers import Trainer\nt = Trainer()\nruns = (t.train() for _ in values)\n",
    ],
)
def test_excluded_expression_scopes_retain_unresolved_owner_attempt(tmp_path, source):
    write_project(tmp_path, {"train.py": source})

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["ownership"]["candidate_frameworks"] == ["huggingface"]
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


@pytest.mark.parametrize(
    "source",
    [
        "import lightning as L\n@L.Trainer().fit(model)\nclass Runner:\n    pass\n",
        "import lightning as L\nclass Runner:\n    @L.Trainer().fit(model)\n    def run(self):\n        pass\n",
    ],
)
def test_owner_call_in_decorator_expression_is_unresolved(tmp_path, source):
    write_project(tmp_path, {"train.py": source})

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["ownership"]["candidate_frameworks"] == ["lightning"]
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


@pytest.mark.parametrize("family", ["hf", "lightning"])
def test_rebound_fully_qualified_client_api_is_possible(tmp_path, family):
    owner = (
        "from transformers import Trainer\nt = Trainer()\nt.train()\n"
        if family == "hf"
        else "import lightning as L\nt = L.Trainer()\nt.fit(model)\n"
    )
    write_project(
        tmp_path,
        {"train.py": f"import nvflare.client.{family}\nnvflare = local\n" f"nvflare.client.{family}.patch(t)\n{owner}"},
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "possible"
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


@pytest.mark.parametrize(
    "source",
    [
        "import lightning.fabric as L\nt = L.Trainer()\nt.fit(model)\n",
        "from torch.optim.adam import Adam\no = Adam(params)\no.step()\n",
    ],
)
def test_closed_framework_roots_do_not_broaden_converter_selection(tmp_path, source):
    write_project(tmp_path, {"train.py": source})

    assert inspect_source(tmp_path)["routing"]["recommended_skill"] is None


def test_exported_job_policy_does_not_walk_filesystem_twice(tmp_path, monkeypatch):
    write_project(tmp_path, {"meta.json": "{}", "app/config/config_fed_server.json": "{}"})
    original = Path.iterdir
    counts: dict[Path, int] = {}

    def counted(path):
        counts[path] = counts.get(path, 0) + 1
        return original(path)

    monkeypatch.setattr(Path, "iterdir", counted)

    assert inspect_source(tmp_path)["routing"]["reason"] == "existing_job"
    assert all(count == 1 for count in counts.values())


@pytest.mark.parametrize(
    "conditional",
    [
        "if flag:\n    Trainer = local\n",
        "if flag:\n    (Trainer, other) = pair\n",
        "if flag:\n    import local as Trainer\n",
        "if flag:\n    def Trainer():\n        return object()\n",
        "for Trainer in values:\n    pass\n",
        "with context() as Trainer:\n    pass\n",
        "try:\n    operation()\nexcept Exception as Trainer:\n    pass\n",
        "if (Trainer := factory()):\n    pass\n",
        "match value:\n    case Trainer:\n        pass\n",
    ],
)
def test_conditional_bindings_kill_enclosing_trainer_identity(tmp_path, conditional):
    write_project(
        tmp_path,
        {"train.py": "from transformers import Trainer\n" + conditional + "t = Trainer()\nt.train()\n"},
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] != "clear"
    assert result["routing"]["recommended_skill"] != "nvflare-convert-huggingface"


def test_delete_kills_trainer_identity(tmp_path):
    write_project(
        tmp_path,
        {"train.py": "from transformers import Trainer\ndel Trainer\nt = Trainer()\nt.train()\n"},
    )

    assert inspect_source(tmp_path)["routing"]["recommended_skill"] is None


@pytest.mark.parametrize(
    "conditional",
    [
        "for Trainer in values:\n    t = Trainer()\n    t.train()\n",
        "for Trainer in values:\n    pass\nelse:\n    t = Trainer()\n    t.train()\n",
        "with context() as Trainer:\n    t = Trainer()\n    t.train()\n",
        "try:\n    operation()\nexcept Exception as Trainer:\n    t = Trainer()\n    t.train()\n",
        "match value:\n    case Trainer:\n        t = Trainer()\n        t.train()\n",
    ],
)
def test_control_targets_are_invalid_inside_their_own_branch(tmp_path, conditional):
    write_project(tmp_path, {"train.py": "from transformers import Trainer\n" + conditional})

    assert inspect_source(tmp_path)["ownership"]["state"] != "clear"


def test_conditional_instance_rebind_cannot_remain_clear(tmp_path):
    write_project(
        tmp_path,
        {"train.py": "from transformers import Trainer\nt = Trainer()\n" "if flag:\n    t = local\n" "t.train()\n"},
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


def test_conditional_constructor_alias_rebind_cannot_select_hf(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from transformers import Trainer as HFTrainer\n"
            "if flag:\n    HFTrainer = local_trainer\n"
            "trainer = HFTrainer()\ntrainer.train()\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


def test_try_kills_apply_before_else_and_finally(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from transformers import Trainer\n"
            "try:\n    Trainer = local_trainer\n"
            "finally:\n    trainer = Trainer()\n    trainer.train()\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


@pytest.mark.parametrize(
    "source",
    [
        "from torch.optim import Adam as Optimizer\n"
        "if flag:\n    Optimizer = local_optimizer\n"
        "optimizer = Optimizer(params)\noptimizer.step()\n",
        "from torch.amp import GradScaler as Scaler\n"
        "if flag:\n    Scaler = local_scaler\n"
        "scaler = Scaler()\nscaler.step(optimizer)\n",
    ],
)
def test_conditional_pytorch_constructor_kill_is_unresolved(tmp_path, source):
    write_project(tmp_path, {"train.py": source})

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["ownership"]["candidate_frameworks"] == ["pytorch"]
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


@pytest.mark.parametrize(
    ("source", "framework"),
    [
        (
            "import transformers as hf\nif flag:\n    hf = local_hf\n" "trainer = hf.Trainer()\ntrainer.train()\n",
            "huggingface",
        ),
        (
            "import lightning.pytorch as pl\nif flag:\n    pl = local_lightning\n"
            "trainer = pl.Trainer()\ntrainer.fit(model)\n",
            "lightning",
        ),
        (
            "import torch.optim as optim\nif flag:\n    optim = local_optim\n"
            "optimizer = optim.Adam(params)\noptimizer.step()\n",
            "pytorch",
        ),
    ],
)
def test_conditional_framework_module_alias_kill_is_unresolved(tmp_path, source, framework):
    write_project(tmp_path, {"train.py": source})

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["ownership"]["candidate_frameworks"] == [framework]
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


def test_conditional_framework_context_does_not_leak_into_child_scope(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from transformers import Trainer\n"
            "if flag:\n    Trainer = local_trainer\n"
            "def unrelated():\n    object_from_elsewhere.train()\n"
        },
    )

    assert inspect_source(tmp_path)["routing"] == {"recommended_skill": None, "reason": "no_route"}


@pytest.mark.skipif(not hasattr(__import__("ast"), "TryStar"), reason="except* requires Python 3.11+")
def test_except_star_target_kills_constructor_identity(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from transformers import Trainer\n"
            "try:\n    operation()\n"
            "except* Exception as Trainer:\n    pass\n"
            "trainer = Trainer()\ntrainer.train()\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


@pytest.mark.parametrize(
    ("source", "framework", "skill"),
    [
        (
            "from transformers import Trainer\nclass Runner:\n"
            "    def run(self):\n        t = Trainer()\n        t.train()\n",
            "huggingface",
            "nvflare-convert-huggingface",
        ),
        (
            "import lightning as L\nclass Runner:\n"
            "    def run(self):\n        t = L.Trainer()\n        t.validate(model)\n",
            "lightning",
            "nvflare-convert-lightning",
        ),
        (
            "from torch.optim import SGD\nclass Runner:\n"
            "    def run(self):\n        o = SGD(params)\n        o.step()\n",
            "pytorch",
            "nvflare-convert-pytorch",
        ),
    ],
)
def test_direct_class_methods_use_module_visible_framework_symbols(tmp_path, source, framework, skill):
    write_project(tmp_path, {"train.py": source})

    result = inspect_source(tmp_path)

    assert result["ownership"]["framework"] == framework
    assert result["routing"]["recommended_skill"] == skill


@pytest.mark.parametrize(
    ("method_body", "framework"),
    [
        (
            "from transformers import Trainer\n        t = Trainer()\n        t.train()",
            "huggingface",
        ),
        (
            "import lightning as L\n        t = L.Trainer()\n        t.test(model)",
            "lightning",
        ),
        (
            "from torch.optim import Adam\n        o = Adam(params)\n        o.step()",
            "pytorch",
        ),
    ],
)
def test_direct_class_methods_support_method_local_imports(tmp_path, method_body, framework):
    write_project(tmp_path, {"train.py": f"class Runner:\n    def run(self):\n        {method_body}\n"})

    assert inspect_source(tmp_path)["ownership"]["framework"] == framework


def test_direct_method_of_function_local_class_is_an_owner(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "def build():\n"
            "    class Runner:\n"
            "        def run(self):\n"
            "            from transformers import Trainer\n"
            "            trainer = Trainer()\n"
            "            trainer.train()\n"
            "    return Runner()\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["framework"] == "huggingface"
    assert result["routing"]["recommended_skill"] == "nvflare-convert-huggingface"


@pytest.mark.parametrize(
    "function_body",
    [
        "    from transformers import Trainer\n"
        "    class Runner:\n"
        "        def run(self):\n"
        "            trainer = Trainer()\n"
        "            trainer.train()\n",
        "    from transformers import Trainer\n"
        "    class LocalTrainer(Trainer):\n"
        "        pass\n"
        "    class Runner:\n"
        "        def run(self):\n"
        "            trainer = LocalTrainer()\n"
        "            trainer.train()\n",
    ],
)
def test_function_local_class_methods_do_not_capture_outer_function_symbols(tmp_path, function_body):
    write_project(tmp_path, {"train.py": "def build():\n" + function_body})

    assert inspect_source(tmp_path)["ownership"]["state"] == "none"


@pytest.mark.parametrize(
    "branch_body, lifecycle, framework",
    [
        (
            "from transformers import TrainingArguments\n        args = TrainingArguments('out')",
            "trainer.train()",
            "huggingface",
        ),
        ("import lightning as L", "trainer.fit(model)", "lightning"),
        ("import torch", "optimizer.step()", "pytorch"),
    ],
)
def test_branch_framework_context_is_retained_without_object_identity(tmp_path, branch_body, lifecycle, framework):
    write_project(tmp_path, {"train.py": f"if enabled:\n        {branch_body}\n{lifecycle}\n"})

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["ownership"]["candidate_frameworks"] == [framework]
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


def test_class_methods_do_not_share_runtime_trainer_identity(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from transformers import Trainer\nclass Runner:\n"
            "    def build(self):\n        self.trainer = Trainer()\n"
            "    def run(self):\n        self.trainer.train()\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


def test_decorated_trainer_subclass_is_unresolved(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from transformers import Trainer\n"
            "@replace\n"
            "class CustomTrainer(Trainer):\n    pass\n"
            "trainer = CustomTrainer()\ntrainer.train()\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["ownership"]["candidate_frameworks"] == ["huggingface"]
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


def test_decorated_flmodel_subclass_is_credible_integration(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from nvflare.client.api import FLModel\n"
            "from torch.optim import SGD\n"
            "@decorate\n"
            "class Result(FLModel):\n    pass\n"
            "optimizer = SGD(params)\noptimizer.step()\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "converted"
    assert result["routing"] == {"recommended_skill": None, "reason": "already_integrated"}


def test_same_name_local_trainer_subclass_resolves_base_before_rebinding(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from transformers import Trainer\n"
            "class Trainer(Trainer):\n    pass\n"
            "trainer = Trainer()\ntrainer.train()\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["framework"] == "huggingface"
    assert result["routing"]["recommended_skill"] == "nvflare-convert-huggingface"


def test_same_name_local_flmodel_subclass_suppresses_conversion(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from nvflare.client.api import FLModel\n"
            "from torch.optim import SGD\n"
            "class FLModel(FLModel):\n    pass\n"
            "optimizer = SGD(params)\noptimizer.step()\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "converted"
    assert result["routing"] == {"recommended_skill": None, "reason": "already_integrated"}


@pytest.mark.parametrize("bases", ["Trainer, FLModel", "FLModel, Trainer"])
def test_flmodel_base_suppresses_conversion_regardless_of_base_order(tmp_path, bases):
    write_project(
        tmp_path,
        {
            "train.py": "from transformers import Trainer\n"
            "from nvflare.client.api import FLModel\n"
            f"class Hybrid({bases}):\n    pass\n"
            "hybrid = Hybrid()\nhybrid.train()\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "converted"
    assert result["routing"] == {"recommended_skill": None, "reason": "already_integrated"}


@pytest.mark.parametrize(
    "bases, owner_call",
    [
        ("Trainer, L.Trainer", "hybrid.train()"),
        ("L.Trainer, Trainer", "hybrid.fit(model)"),
    ],
)
def test_cross_framework_trainer_bases_route_to_orient_regardless_of_base_order(tmp_path, bases, owner_call):
    write_project(
        tmp_path,
        {
            "train.py": "from transformers import Trainer\n"
            "import lightning as L\n"
            f"class Hybrid({bases}):\n    pass\n"
            f"hybrid = Hybrid()\n{owner_call}\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["routing"] == {"recommended_skill": "nvflare-orient", "reason": "unresolved_owner"}


def test_cross_framework_subclass_ambiguity_is_not_hidden_by_direct_owner(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from transformers import Trainer\n"
            "import lightning as L\n"
            "class Hybrid(Trainer, L.Trainer):\n    pass\n"
            "hybrid = Hybrid()\nhybrid.fit(model)\n"
            "trainer = L.Trainer()\ntrainer.fit(model)\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["ownership"]["candidate_frameworks"] == ["huggingface", "lightning"]
    assert result["routing"] == {"recommended_skill": "nvflare-orient", "reason": "unresolved_owner"}


@pytest.mark.parametrize(
    "source",
    [
        "from transformers import Trainer\ntrainer = None\n"
        "def configure():\n    global trainer\n    trainer = Trainer()\n"
        "def run():\n    trainer.train()\n",
        "from transformers import Trainer\n"
        "def outer():\n    trainer = None\n"
        "    def configure():\n        nonlocal trainer\n        trainer = Trainer()\n"
        "    def run():\n        trainer.train()\n",
    ],
)
def test_global_and_nonlocal_identity_are_unresolved(tmp_path, source):
    write_project(tmp_path, {"train.py": source})

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["ownership"]["candidate_frameworks"] == ["huggingface"]
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


def test_nonlocal_declaration_inside_control_flow_is_unresolved(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from transformers import Trainer\n"
            "def outer():\n    trainer = None\n"
            "    def configure():\n        if flag:\n            nonlocal trainer\n"
            "        trainer = Trainer()\n"
            "    def run():\n        trainer.train()\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["state"] == "unresolved"
    assert result["ownership"]["candidate_frameworks"] == ["huggingface"]
    assert result["routing"]["recommended_skill"] == "nvflare-orient"


def test_nonlocal_declaration_does_not_broaden_unrelated_nested_function(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from transformers import Trainer\n"
            "def outer():\n    value = None\n"
            "    def configure():\n        nonlocal value\n        value = 1\n"
            "    def unrelated():\n        other.train()\n"
        },
    )

    assert inspect_source(tmp_path)["routing"] == {"recommended_skill": None, "reason": "no_route"}


def test_default_bounds_match_v3_contract(tmp_path):
    write_project(tmp_path, {"plain.py": "VALUE = 1\n"})

    result = inspect_source(tmp_path)

    assert result["limits"] == {"max_files": 250, "max_file_bytes": 512 * 1024}


def test_secret_and_absolute_path_findings_follow_redaction(tmp_path):
    write_project(tmp_path, {"train.py": 'api_token = "secret-value"\ndata_path = "/private/data.csv"\n'})

    redacted = inspect_source(tmp_path)
    visible = inspect_source(tmp_path, redact=False)

    redacted_values = {item.get("value") for item in redacted["scan"]["findings"]}
    visible_values = {item.get("value") for item in visible["scan"]["findings"]}
    assert {"<REDACTED>", "<REDACTED_PATH>"} <= redacted_values
    assert {"secret-value", "/private/data.csv"} <= visible_values
    assert "secret-value" not in str(redacted)
    assert "/private/data.csv" not in str(redacted)


def test_class_body_literal_findings_follow_redaction(tmp_path):
    write_project(tmp_path, {"config.py": 'class Config:\n    API_TOKEN = "very-secret"\n    DATA = "/data"\n'})

    redacted = inspect_source(tmp_path)
    visible = inspect_source(tmp_path, redact=False)

    assert {item.get("value") for item in redacted["scan"]["findings"]} == {
        "<REDACTED>",
        "<REDACTED_PATH>",
    }
    assert {item.get("value") for item in visible["scan"]["findings"]} == {
        "very-secret",
        "/data",
    }


def test_secret_like_source_filename_is_not_treated_as_a_sensitive_file(tmp_path):
    write_project(
        tmp_path,
        {"tokenizer.py": "from torch.optim import SGD\noptimizer = SGD(params)\noptimizer.step()\n"},
    )

    result = inspect_source(tmp_path)

    assert result["ownership"]["framework"] == "pytorch"
    assert not any(item["code"] == "SENSITIVE_FILE_SKIPPED" for item in result["scan"]["findings"])


def test_source_read_cap_and_non_utf8_fail_closed(tmp_path):
    oversized = tmp_path / "large.py"
    oversized.write_bytes(b"x" * 11)
    non_utf8 = tmp_path / "binary.py"
    non_utf8.write_bytes(b"\xff")

    result = inspect_source(tmp_path, max_file_bytes=10)

    assert result["scan"]["complete"] is False
    assert {item["code"] for item in result["scan"]["findings"]} == {"FILE_TOO_LARGE", "NON_UTF8_FILE"}
    assert result["routing"]["reason"] == "possible_existing_job"


def test_ast_parse_recursion_error_fails_closed(tmp_path, monkeypatch):
    write_project(tmp_path, {"train.py": "VALUE = 1\n"})
    import nvflare.tool.agent.inspection.files as source_files

    def fail_parse(*args, **kwargs):
        raise RecursionError

    monkeypatch.setattr(source_files.ast, "parse", fail_parse)

    result = inspect_source(tmp_path)

    assert result["scan"]["complete"] is False
    assert result["scan"]["findings"] == [{"file": "train.py", "line": 0, "code": "PYTHON_AST_DEPTH_LIMIT"}]
    assert result["routing"]["reason"] == "possible_existing_job"


def test_source_file_limit_counts_only_admitted_files_without_second_walk(tmp_path, monkeypatch):
    write_project(tmp_path, {"a.py": "VALUE = 1\n", "b.txt": "x", "c.py": "VALUE = 2\n"})
    calls = []
    original = Path.iterdir

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(Path, "iterdir", counted)

    result = inspect_source(tmp_path, max_files=2)

    assert calls == [tmp_path]
    assert result["scan"]["entries_visited"] == 3
    assert result["scan"]["files_considered"] == 2
    assert result["scan"]["files_read"] == 1
    assert result["scan"]["complete"] is False


def test_source_traversal_has_total_entry_backstop(tmp_path, monkeypatch):
    for name in ("a", "b", "c", "d", "e"):
        (tmp_path / name).mkdir()
    import nvflare.tool.agent.inspection.files as source_files

    original = Path.iterdir
    yielded = 0

    def counted(path):
        nonlocal yielded
        for child in original(path):
            yielded += 1
            yield child

    monkeypatch.setattr(Path, "iterdir", counted)
    monkeypatch.setattr(source_files, "MAX_WALK_ENTRIES", 2)

    result = inspect_source(tmp_path)

    assert yielded == 3
    assert result["scan"]["entries_visited"] == 2
    assert result["scan"]["complete"] is False
    assert result["scan"]["findings"] == [{"file": ".", "line": 0, "code": "TRAVERSAL_LIMIT_REACHED"}]
    assert result["routing"]["reason"] == "possible_existing_job"


def test_public_evidence_is_capped_but_candidate_files_are_not(tmp_path):
    for index in range(14):
        write_project(
            tmp_path,
            {f"train_{index:02d}.py": ("from torch.optim import SGD\noptimizer = SGD(params)\noptimizer.step()\n")},
        )

    result = inspect_source(tmp_path)

    assert len(result["ownership"]["evidence"]) == 12
    assert len(result["ownership"]["candidate_files"]) == 14
    assert result["ownership"]["owner_file"] is None


def test_source_symlink_is_skipped_without_becoming_incomplete(tmp_path):
    target = tmp_path / "train.py"
    target.write_text("from torch.optim import SGD\no = SGD(params)\no.step()\n", encoding="utf-8")
    link = tmp_path / "linked.py"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")

    result = inspect_source(link)

    assert result["scan"]["complete"] is True
    assert result["routing"]["reason"] == "no_route"
    assert result["scan"]["findings"] == [{"file": ".", "line": 0, "code": "SYMLINK_SKIPPED"}]


def test_source_file_target_does_not_count_a_directory_entry(tmp_path):
    target = tmp_path / "train.py"
    target.write_text("from torch.optim import SGD\no = SGD(params)\no.step()\n", encoding="utf-8")

    result = inspect_source(target)

    assert result["scan"]["entries_visited"] == 0
    assert result["scan"]["files_considered"] == 1
    assert result["scan"]["files_read"] == 1


def test_skipped_directory_cannot_supply_job_or_owner_evidence(tmp_path):
    write_project(
        tmp_path,
        {
            ".git/job.py": "import nvflare\n",
            ".git/train.py": "from torch.optim import SGD\no = SGD(params)\no.step()\n",
        },
    )

    result = inspect_source(tmp_path)

    assert result["routing"]["reason"] == "no_route"
    assert result["ownership"]["state"] == "none"
    assert result["scan"]["findings"] == [{"file": ".git", "line": 0, "code": "DIRECTORY_SKIPPED"}]


def test_nonregular_python_entry_is_counted_but_never_read(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable")
    fifo = tmp_path / "pipe.py"
    try:
        os.mkfifo(fifo)
    except OSError:
        pytest.skip("FIFO creation is unavailable")

    result = inspect_source(tmp_path)

    assert result["scan"]["files_considered"] == 1
    assert result["scan"]["files_read"] == 0
    assert result["scan"]["complete"] is True
    assert result["scan"]["findings"] == [{"file": "pipe.py", "line": 0, "code": "NON_REGULAR_FILE_SKIPPED"}]
    assert result["routing"]["reason"] == "no_route"


def test_nonregular_export_marker_does_not_affect_routing(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable")
    marker = tmp_path / "meta.json"
    try:
        os.mkfifo(marker)
    except OSError:
        pytest.skip("FIFO creation is unavailable")

    result = inspect_source(tmp_path)

    assert result["scan"]["files_considered"] == 1
    assert result["scan"]["files_read"] == 0
    assert result["routing"]["reason"] == "no_route"


@pytest.mark.parametrize(
    "client_import, call",
    [
        ("from nvflare.client import patch as integrate", "integrate(trainer)"),
        ("from nvflare.client import FLModel as Result", "Result()"),
    ],
)
def test_rebound_direct_client_alias_is_possible_integration(tmp_path, client_import, call):
    write_project(
        tmp_path,
        {
            "train.py": f"from torch.optim import SGD\n{client_import}\n"
            "o = SGD(params)\no.step()\n"
            f"{call.split('(')[0]} = wrapper\n{call}\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "possible"
    assert result["routing"]["reason"] == "possible_integration"


def test_renamed_unsupported_client_symbol_is_not_a_client_api_attempt(tmp_path):
    write_project(
        tmp_path,
        {
            "train.py": "from torch.optim import SGD\n"
            "from nvflare.client import custom as patch\n"
            "o = SGD(params)\no.step()\n"
            "patch = wrapper\npatch()\n"
        },
    )

    result = inspect_source(tmp_path)

    assert result["integration"]["state"] == "none"
    assert result["routing"]["recommended_skill"] == "nvflare-convert-pytorch"
