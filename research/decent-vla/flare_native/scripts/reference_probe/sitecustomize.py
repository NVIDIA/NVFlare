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

"""Opt-in instrumentation of the unchanged reference runtime for parity runs."""

import os

if os.environ.get("DVLA_REFERENCE_PROBE") == "1":
    import json
    import random
    from pathlib import Path

    import numpy as np
    import torch

    import decent_vla.runtime.builders as builders
    import decent_vla.runtime.round_loop as rounds

    # Resolve all reference adapters from the parent source tree. The native
    # package has no CLI/plugin entry points in its installed metadata.
    def reference_registry():
        from decent_vla.adapters._mock.backend import MockBackend
        from decent_vla.adapters.groot.backend import GrootBackend
        from decent_vla.adapters.lerobot.backend import LeRobotBackend
        from decent_vla.benchmarks._mock import MockBenchmark
        from decent_vla.benchmarks.libero import LIBEROBenchmark
        from decent_vla.core.registry import Registry
        from decent_vla.strategies import FedAdam, FedAvg, FedProx

        return Registry(
            backends={"mock": MockBackend, "lerobot": LeRobotBackend, "groot": GrootBackend},
            benchmarks={"mock": MockBenchmark, "libero": LIBEROBenchmark},
            strategies={"fedavg": FedAvg, "fedadam": FedAdam, "fedprox": FedProx},
        )

    builders.discover = reference_registry

    original_factory = builders.build_backend_factory

    def build_factory(cfg, device):
        factory, cls = original_factory(cfg, device)

        def seeded_factory(*args, **kwargs):
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)
            if cfg.backend.extra.get("deterministic", False):
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            return factory(*args, **kwargs)

        return seeded_factory, cls

    builders.build_backend_factory = build_factory

    original_rounds = rounds.run_rounds

    def recorded_rounds(*args, **kwargs):
        outcome = original_rounds(*args, **kwargs)
        # Only the server invokes run_rounds. Client subprocesses never write here.
        path = Path(os.environ["DVLA_REFERENCE_CURVE"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(outcome.per_round_metrics, indent=2))
        return outcome

    rounds.run_rounds = recorded_rounds

    # Python suppresses sitecustomize import errors. A launcher preflight checks
    # this marker so a broken probe cannot silently run an unseeded reference.
    os.environ["DVLA_REFERENCE_PROBE_ACTIVE"] = "1"
