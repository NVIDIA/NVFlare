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
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, field_validator

from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.ccwf.ccwf_job import CCWFJob, CrossSiteEvalConfig, SwarmClientConfig, SwarmServerConfig
from nvflare.app_common.ccwf.comps.simple_model_shareable_generator import SimpleModelShareableGenerator
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.spec import Recipe
from nvflare.recipe.utils import validate_initial_ckpt


class _SwarmValidator(BaseModel):
    initial_ckpt: Optional[str] = None

    @field_validator("initial_ckpt")
    @classmethod
    def validate_initial_ckpt(cls, v):
        if v is not None:
            validate_initial_ckpt(v)
        return v

    model_config = {"arbitrary_types_allowed": True}


def _instantiate_model_from_dict(model_config: Dict[str, Any]) -> Any:
    """Instantiate a model from dict config.

    Args:
        model_config: Dict with 'path' (required) and 'args' (optional) keys.
            Example: {"path": "my_module.MyModel", "args": {"num_classes": 10}}

    Returns:
        Instantiated model

    Raises:
        ValueError: If 'path' key is missing or model cannot be instantiated.
    """
    if "path" not in model_config:
        raise ValueError("model_config dict must contain 'path' key with the model class path")

    class_path = model_config["path"]
    model_args = model_config.get("args", {})

    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        return model_class(**model_args)
    except Exception as e:
        raise ValueError(f"Failed to instantiate model from '{class_path}': {str(e)}") from e


class BaseSwarmLearningRecipe(Recipe):
    """Base recipe for Swarm Learning (framework-agnostic).

    Args:
        name: Name of the federated learning job.
        server_config: Swarm server configuration.
        client_config: Swarm client configuration.
        cse_config: Optional cross-site evaluation configuration.
        job: Optional pre-created CCWFJob. If None, a new one is created.
            Subclasses may create the job early to add files before building configs.
    """

    def __init__(
        self,
        name: str,
        server_config: SwarmServerConfig,
        client_config: SwarmClientConfig,
        cse_config: CrossSiteEvalConfig = None,
        job: CCWFJob = None,
    ):
        if job is None:
            job = CCWFJob(name=name)
        job.add_swarm(
            server_config=server_config,
            client_config=client_config,
            cse_config=cse_config,
        )
        Recipe.__init__(self, job)


class SimpleSwarmLearningRecipe(BaseSwarmLearningRecipe):
    """A simple recipe for Swarm Learning with PyTorch models.

    Args:
        name: Name of the federated learning job.
        model: PyTorch model to use as the initial model. Can be:
            - An nn.Module instance (e.g., MyModel())
            - A dict config: {"path": "module.ClassName", "args": {"param": value}}
        initial_ckpt: Path to a pre-trained checkpoint file (.pt, .pth). Can be:
            - Relative path: file will be bundled into the job's custom/ directory.
            - Absolute path: treated as a server-side path, used as-is at runtime.
        num_rounds: Number of training rounds.
        train_script: Path to the training script.
        train_args: Additional arguments for the training script.
        do_cross_site_eval: Whether to perform cross-site evaluation.
        cross_site_eval_timeout: Timeout for cross-site evaluation.

    Example:
        Using nn.Module instance:

        ```python
        recipe = SimpleSwarmLearningRecipe(
            name="swarm_job",
            model=MyModel(),
            num_rounds=5,
            train_script="train.py",
        )
        ```

        Using dict config:

        ```python
        recipe = SimpleSwarmLearningRecipe(
            name="swarm_job",
            model={"path": "my_module.MyModel", "args": {"num_classes": 10}},
            num_rounds=5,
            train_script="train.py",
        )
        ```
    """

    def __init__(
        self,
        name: str,
        model: Union[Any, Dict[str, Any]],
        num_rounds: int,
        train_script: str,
        initial_ckpt: Optional[str] = None,
        train_args: dict = None,
        do_cross_site_eval: bool = False,
        cross_site_eval_timeout: float = 300,
    ):
        _SwarmValidator(initial_ckpt=initial_ckpt)

        # Handle dict-based model config
        if isinstance(model, dict):
            model_instance = _instantiate_model_from_dict(model)
        else:
            model_instance = model

        aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS)
        if do_cross_site_eval:
            cse_config = CrossSiteEvalConfig(eval_task_timeout=cross_site_eval_timeout)
        else:
            cse_config = None

        if not train_args:
            train_args = {}
        else:
            # Validate train_args doesn't conflict with ScriptRunner reserved parameters
            reserved_keys = {"script", "launch_external_process", "command", "framework"}
            conflicts = set(train_args.keys()) & reserved_keys
            if conflicts:
                raise ValueError(f"train_args contains reserved keys that conflict with ScriptRunner: {conflicts}")

        # Create job early so prepare_initial_ckpt can bundle files into it
        from nvflare.recipe.utils import prepare_initial_ckpt

        job = CCWFJob(name=name)
        ckpt_path = prepare_initial_ckpt(initial_ckpt, job)

        server_config = SwarmServerConfig(num_rounds=num_rounds)
        client_config = SwarmClientConfig(
            executor=ScriptRunner(script=train_script, **train_args),
            aggregator=aggregator,
            persistor=PTFileModelPersistor(model=model_instance, source_ckpt_file_full_name=ckpt_path),
            shareable_generator=SimpleModelShareableGenerator(),
        )

        BaseSwarmLearningRecipe.__init__(self, name, server_config, client_config, cse_config, job=job)
