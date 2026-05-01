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

import logging
import os
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, field_validator

from nvflare.apis.dxo import DataKind
from nvflare.apis.fl_constant import SystemVarName
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.ccwf.ccwf_job import CCWFJob, CrossSiteEvalConfig, SwarmClientConfig, SwarmServerConfig
from nvflare.app_common.ccwf.comps.simple_model_shareable_generator import SimpleModelShareableGenerator
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.file_pipe import FilePipe
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.spec import Recipe
from nvflare.recipe.utils import validate_ckpt

logger = logging.getLogger(__name__)

_VALID_PIPE_TYPES = ("cell_pipe", "file_pipe")


class _SwarmValidator(BaseModel):
    initial_ckpt: Optional[str] = None

    @field_validator("initial_ckpt")
    @classmethod
    def validate_initial_ckpt(cls, v):
        if v is not None:
            validate_ckpt(v)
        return v

    model_config = {"arbitrary_types_allowed": True}


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


class SwarmLearningRecipe(BaseSwarmLearningRecipe):
    """A simple recipe for Swarm Learning with PyTorch models.

    Args:
        name: Name of the federated learning job.
        model: PyTorch model to use as the initial model. Can be:
            - An nn.Module instance (e.g., MyModel())
            - A dict config: {"class_path": "module.ClassName", "args": {"param": value}}
        num_rounds: Number of training rounds.
        train_script: Path to the training script.
        min_clients: Minimum number of clients required.
        initial_ckpt: Path to a pre-trained checkpoint file (.pt, .pth). Can be:
            - Relative path: file will be bundled into the job's custom/ directory.
            - Absolute path: treated as a server-side path, used as-is at runtime.
        train_args: Additional arguments for the training script.
        do_cross_site_eval: Whether to perform cross-site evaluation. When combined with
            ``launch_external_process=True``, the trained model is loaded from the
            persistor on disk (saved by PTFileModelPersistor after each round).  Two
            limitations apply in that combination:

            1. **Custom persistors**: If your persistor streams models to a remote store
               without supporting local ``get()``, the persistor path returns None and
               CSE falls back to the executor, which also fails for ext-process mode.
               Ensure your persistor's ``get()`` can retrieve the model locally.
            2. **Cross-job evaluation**: CSE against a model trained in a *different* job
               is not supported with ``launch_external_process=True`` because the current
               job's persistor cannot locate the other job's workspace. Use in-process
               mode or copy the trained model into the evaluating job's workspace.
        cross_site_eval_timeout: Timeout for cross-site evaluation.
        launch_external_process: Whether to launch the training script in an external process.
            Defaults to False (in-process execution).
        command: Shell command used to launch the script when launch_external_process=True.
            Defaults to "python3 -u".
        memory_gc_rounds: Run gc.collect() + malloc_trim every N FL rounds on both the trainer
            and aggregator roles. Defaults to 1 (every round) to match legacy behavior where
            gc.collect() was called unconditionally after each trainer submission. Set to 0 to disable.
        cuda_empty_cache: Call torch.cuda.empty_cache() during cleanup. Defaults to False.
        expected_data_kind: The data kind the aggregator expects from clients. Defaults to
            DataKind.WEIGHTS for full-weight FedAvg. Use DataKind.WEIGHT_DIFF when clients
            send parameter deltas (e.g. LoRA adapter diffs with params_transfer_type=DIFF).
        params_transfer_type: How parameters are transferred between client script and NVFlare.
            FULL sends the entire parameter state each round; DIFF sends only the delta.
            Defaults to FULL. Must match the ParamsType used in the training script.
        start_task_timeout: Seconds to wait for the starting client to acknowledge the start
            task. Increase for large models that need time to load. Defaults to 300.
        progress_timeout: Seconds of no progress from any client before the workflow is
            considered stalled. Defaults to 3600.
        max_status_report_interval: Maximum seconds between consecutive status reports from
            a client before it is considered silent. Defaults to 300.
        round_timeout: P2P model transfer ACK budget in seconds — how long the aggregator
            waits for a receiver to acknowledge the model download via tensor streaming.
            The "ACK" includes the full model download, so the hardcoded default of 10s
            in SwarmClientConfig is too short for models larger than ~2GB.  Set higher
            for large models (7B+) where P2P transfer can take minutes.  Does NOT cap
            per-round training time (learn_task_timeout remains unbounded by default).
            Defaults to 3600 (matching progress_timeout).
        pipe_type: Pipe used for communication between the NVFlare client process
            and the external training process when ``launch_external_process=True``.
            Accepted values:

            - ``"cell_pipe"`` *(default)*: ``CellPipe`` with zero-copy tensor
              forwarding — the NVFlare client process relays model tensors without
              loading them into memory (~1 GB RAM for large models).
            - ``"file_pipe"``: ``FilePipe`` backed by a shared directory. The NVFlare
              client process fully loads and re-serializes the model (~2× model size
              in RAM). Use when cell networking is unavailable or for third-party
              integrations that cannot resolve NVFlare cell addresses.

            Ignored when ``launch_external_process=False``.
        pipe_root_path: Base directory for ``FilePipe`` when ``pipe_type="file_pipe"``.
            ``None`` (default) uses ``{WORKSPACE}/{JOB_ID}/{SITE_NAME}``, matching
            the ``sag_cse_ccwf_pt`` reference template. If provided, the path must be
            an absolute path (e.g. ``"/dev/shm/nvflare_pipes"`` for a RAM-backed tmpfs);
            the directory is treated as a runtime path and does not need to exist on the
            machine that builds or exports the job. ``{JOB_ID}/{SITE_NAME}`` is always
            appended so concurrent jobs and sites remain isolated. Ignored for ``"cell_pipe"``.

    Example:
        Using nn.Module instance:

        ```python
        recipe = SwarmLearningRecipe(
            name="swarm_job",
            model=MyModel(),
            min_clients=3,
            num_rounds=5,
            train_script="train.py",
        )
        ```

        Using dict config:

        ```python
        recipe = SwarmLearningRecipe(
            name="swarm_job",
            model={"class_path": "my_module.MyModel", "args": {"num_classes": 10}},
            min_clients=3,
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
        min_clients: int,
        initial_ckpt: Optional[str] = None,
        train_args: dict = None,
        do_cross_site_eval: bool = False,
        cross_site_eval_timeout: float = 300,
        launch_external_process: bool = False,
        command: str = "python3 -u",
        memory_gc_rounds: int = 1,
        cuda_empty_cache: bool = False,
        expected_data_kind: str = DataKind.WEIGHTS,
        params_transfer_type: str = "FULL",
        start_task_timeout: float = 300,
        progress_timeout: float = 3600,
        max_status_report_interval: float = 300,
        round_timeout: float = 3600,
        pipe_type: str = "cell_pipe",
        pipe_root_path: Optional[str] = None,
    ):
        _SwarmValidator(initial_ckpt=initial_ckpt)

        if pipe_type not in _VALID_PIPE_TYPES:
            raise ValueError(f"pipe_type must be one of {_VALID_PIPE_TYPES}, got '{pipe_type}'")

        if pipe_root_path and pipe_type != "file_pipe":
            logger.warning(
                f"pipe_root_path='{pipe_root_path}' is ignored when pipe_type='{pipe_type}' "
                "(only applies to 'file_pipe')"
            )

        if pipe_root_path and pipe_type == "file_pipe":
            if not os.path.isabs(pipe_root_path):
                raise ValueError(f"pipe_root_path must be an absolute path, got '{pipe_root_path}'")

        if pipe_type == "file_pipe" and not launch_external_process:
            logger.warning(
                "pipe_type='file_pipe' has no effect when launch_external_process=False "
                "(in-process mode does not use pipes)"
            )

        task_pipe = None
        if pipe_type == "file_pipe":
            # Append {JOB_ID}/{SITE_NAME} so concurrent jobs and sites on the same
            # machine use isolated pipe directories (resolved at runtime by NVFlare).
            # Format matches the sag_cse_ccwf_pt reference template.
            _job_site_suffix = "/{" + SystemVarName.JOB_ID + "}/{" + SystemVarName.SITE_NAME + "}"
            if pipe_root_path:
                root_path = pipe_root_path + _job_site_suffix
            else:
                root_path = "{" + SystemVarName.WORKSPACE + "}" + _job_site_suffix
            task_pipe = FilePipe(mode=Mode.PASSIVE, root_path=root_path)

        # Handle dict-based model config (recipe accepts class_path; normalize for job API).
        # Pass the dict directly to PTFileModelPersistor so args are preserved in the exported config.
        # The persistor resolves the dict to an nn.Module at runtime via instantiate_class().
        if isinstance(model, dict):
            from nvflare.recipe.utils import recipe_model_to_job_model

            model = recipe_model_to_job_model(model)

        aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=expected_data_kind)
        if do_cross_site_eval:
            cse_config = CrossSiteEvalConfig(eval_task_timeout=cross_site_eval_timeout)
        else:
            cse_config = None

        if not train_args:
            train_args = {}
        else:
            # Validate train_args doesn't conflict with ScriptRunner reserved parameters
            reserved_keys = {
                "script",
                "launch_external_process",
                "command",
                "framework",
                "memory_gc_rounds",
                "cuda_empty_cache",
            }
            conflicts = set(train_args.keys()) & reserved_keys
            if conflicts:
                raise ValueError(f"train_args contains reserved keys that conflict with ScriptRunner: {conflicts}")

        # Create job early so prepare_initial_ckpt can bundle files into it
        from nvflare.recipe.utils import prepare_initial_ckpt

        job = CCWFJob(name=name, min_clients=min_clients)
        ckpt_path = prepare_initial_ckpt(initial_ckpt, job)

        server_config = SwarmServerConfig(
            num_rounds=num_rounds,
            start_task_timeout=start_task_timeout,
            progress_timeout=progress_timeout,
            max_status_report_interval=max_status_report_interval,
            min_clients=min_clients,
        )
        client_config = SwarmClientConfig(
            executor=ScriptRunner(
                script=train_script,
                launch_external_process=launch_external_process,
                command=command,
                memory_gc_rounds=memory_gc_rounds,
                cuda_empty_cache=cuda_empty_cache,
                params_transfer_type=params_transfer_type,
                task_pipe=task_pipe,
                **train_args,
            ),
            aggregator=aggregator,
            persistor=PTFileModelPersistor(model=model, source_ckpt_file_full_name=ckpt_path),
            shareable_generator=SimpleModelShareableGenerator(),
            memory_gc_rounds=memory_gc_rounds,
            cuda_empty_cache=cuda_empty_cache,
            min_responses_required=min_clients,
            learn_task_ack_timeout=round_timeout,
            final_result_ack_timeout=round_timeout,
            # learn_task_timeout intentionally not set — inherits None (unbounded) from
            # SwarmClientConfig default.  Capping per-round training time via round_timeout
            # would regress long-running training on slow hardware or for 70B+ models.
        )

        BaseSwarmLearningRecipe.__init__(self, name, server_config, client_config, cse_config, job=job)
