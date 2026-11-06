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

"""Audit the explicitly chosen final global checkpoint on previously unused final cohorts."""

import argparse
from pathlib import Path

import torch
from hf_model import DossierAuditor, configure, sha256

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.workspace import Workspace
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.widgets.memorization_audit_reporter import MemorizationAuditReporter


class EvaluationWorkspace:
    """Minimal workspace provider for the reporter outside a running federated job."""

    def __init__(self, root):
        self.workspace = Workspace(root_dir=root, site_name="global-evaluation")

    def get_workspace(self):
        return self.workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument(
        "--checkpoint", required=True, help="Use the last FL_global_model.pt, not a model selected on final audits"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configure()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    context = FLContextManager(
        engine=EvaluationWorkspace(str(output)), identity_name="global-evaluation", job_id="final-evaluation"
    ).new_context()
    context.set_prop(FLContextKey.TASK_ID, "final-checkpoint", private=True, sticky=False)
    auditor = DossierAuditor(args.assets, cohort="final")
    report = auditor.audit(FLModel(params=state, current_round=1), None, context)
    report.update(status="completed", round=1, checkpoint_sha256=sha256(args.checkpoint), cohort="final")
    path = MemorizationAuditReporter().report(report, context, stage="global_model")
    if path is None:
        raise RuntimeError("Global audit report could not be written")
    print(path)


if __name__ == "__main__":
    main()
