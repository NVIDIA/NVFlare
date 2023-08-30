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
#change to GNN
import os.path as osp

import torch
from torch_geometric.datasets import Reddit
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from torch_geometric.datasets import PPI
from torch_geometric.data import Batch
from torch_geometric.nn import GraphSAGE

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier


class GraphSageValidator(Executor):
    def __init__(self, data_path="~/data", validate_task_name=AppConstants.TASK_VALIDATION):
        super().__init__()

        self._validate_task_name = validate_task_name
        # Preparing the dataset for testing.
        # Evaluation loaders (one datapoint corresponds to a graph)
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
        train_dataset = PPI(path, split='train')
        val_dataset = PPI(path, split='val')
        test_dataset = PPI(path, split='test')

        self._train_loader = DataLoader(train_dataset, batch_size=2)
        self._val_loader = DataLoader(val_dataset, batch_size=2)
        self._test_loader = DataLoader(test_dataset, batch_size=2)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # Setup the model
        self.model = GraphSAGE(
            in_channels=train_dataset.num_features,
            hidden_channels=64,
            num_layers=2,
            out_channels=64,
            ).to(self.device)


    @torch.no_grad()   
    def encode(self, loader):
        self.model.eval()

        xs, ys = [], []
        for data in loader:
            data = data.to(self.device)
            xs.append(self.model(data.x, data.edge_index).cpu())
            ys.append(data.y.cpu())
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)
    

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self._validate_task_name:
            model_owner = "?"
            try:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

                # Get validation accuracy
                val_accuracy = self._validate(weights, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(
                    fl_ctx,
                    f"Accuracy when validating {model_owner}'s model on"
                    f" {fl_ctx.get_identity_name()}"
                    f"s data: {val_accuracy}",
                )

                dxo = DXO(data_kind=DataKind.METRICS, data={"val_acc": val_accuracy})
                return dxo.to_shareable()
            except:
                self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)


    @torch.no_grad()
    def _validate(self, weights, abort_signal):
        self.model.load_state_dict(weights)

        x, y = self.encode(self._train_loader)

        clf = MultiOutputClassifier(SGDClassifier(loss='log', penalty='l2'))
        clf.fit(x, y)

        train_f1 = f1_score(y, clf.predict(x), average='micro')

        # Evaluate on validation set:
        x, y = self.encode(self._val_loader)
        val_f1 = f1_score(y, clf.predict(x), average='micro')

        # Evaluate on test set:
        x, y = self.encode(self._test_loader)
        test_f1 = f1_score(y, clf.predict(x), average='micro')

        return train_f1, val_f1, test_f1
