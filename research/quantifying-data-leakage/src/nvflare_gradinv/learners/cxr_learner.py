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

import copy
import os

import monai
import numpy as np
import torch
from monai.data import CacheDataset, load_decathlon_datalist
from monai.networks.nets.torchvision_fc import TorchVisionFCModel
from monai.transforms import (
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RepeatChanneld,
    Resized,
    ToNumpyd,
)
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType
from nvflare.app_opt.pt.fedproxloss import PTFedProxLoss


class CXRLearner(Learner):
    def __init__(
        self,
        data_root: str = "./dataset",
        dataset_json: str = "./dataset.json",
        aggregation_epochs: int = 1,
        lr: float = 1e-2,
        fedproxloss_mu: float = 0.0,
        analytic_sender_id: str = "analytic_sender",
        batch_size: int = 64,
        num_workers: int = 0,
        train_set: str = "training",
        valid_set: str = "validation",
        test_set: str = "testing",
        model_name: str = "resnet18",
        num_class: int = 2,
        cache_rate: float = 1.0,
        seed: int = 0,
    ):
        """Simple CXR Trainer

        Args:
            data_root: root directory for data
            dataset_json: JSON data list specifying the train, validation, and test sets.
            aggregation_epochs: the number of training epochs for a round. Defaults to 1.
            lr: local learning rate. Float number. Defaults to 1e-2.
            fedproxloss_mu: weight for FedProx loss. Float number. Defaults to 0.0 (no FedProx).
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component.
                If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
            batch_size: batch size for training and validation.
            num_workers: number of workers for data loaders.
            train_set: name of train data list. Defaults to "training".
            valid_set: name of train data list. Defaults to "validation".
            test_set: name of train data list. Defaults to "testing".
            model_name: name of torchvision model compatible with MONAI's TorchVisionFCModel class. Defaults to "resnet18".
            num_class: Number of prediction classes. Defaults to 2.
            cache_rate: Cache rate used in CacheDataset.
            seed: Seed used for deterministic training.

        Returns:
            a Shareable with the model updates, validation scores, or the best local model depending on the specified task.
        """
        super().__init__()
        # trainer init happens at the very beginning, only the basic info regarding the trainer is set here
        # the actual run has not started at this point

        # some input checks
        if isinstance(data_root, str):
            if not os.path.isdir(data_root):
                raise ValueError(f"`data_root` directory does not exist at {data_root}")
        else:
            raise ValueError(f"Expected `data_root` of type `str` but received type {type(data_root)}")
        if isinstance(dataset_json, str):
            if not os.path.isfile(dataset_json):
                raise ValueError(f"`dataset_json` file does not exist at {dataset_json}")
        else:
            raise ValueError(f"Expected `dataset_json` of type `str` but received type {type(dataset_json)}")

        self.seed = seed
        self.data_root = data_root
        self.dataset_json = dataset_json
        self.aggregation_epochs = aggregation_epochs
        self.lr = lr
        self.fedproxloss_mu = fedproxloss_mu
        self.best_acc = 0.0
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.model_name = model_name
        self.num_class = num_class
        self.cache_rate = cache_rate

        self.writer = None
        self.analytic_sender_id = analytic_sender_id

        # Epoch counter
        self.epoch_of_start_time = 0
        self.epoch_global = 0

        # following will be created in initialize() or later
        self.app_root = None
        self.client_id = None
        self.local_model_file = None
        self.best_local_model_file = None
        self.writer = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.criterion_prox = None
        self.transform_train = None
        self.transform_valid = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.global_weights = None
        self.current_round = None

    def initialize(self, parts: dict, fl_ctx: FLContext):
        # when the run starts, this is where the actual settings get initialized for trainer

        if self.seed is not None:
            self.logger.info(f"Use deterministic training with seed={self.seed}")
            monai.utils.misc.set_determinism(seed=self.seed, use_deterministic_algorithms=True)

        # Set the paths according to fl_ctx
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        fl_args = fl_ctx.get_prop(FLContextKey.ARGS)
        self.client_id = fl_ctx.get_identity_name()
        self.log_info(
            fl_ctx,
            f"Client {self.client_id} initialized at \n {self.app_root} \n with args: {fl_args}",
        )

        self.local_model_file = os.path.join(self.app_root, "local_model.pt")
        self.best_local_model_file = os.path.join(self.app_root, "best_local_model.pt")

        # Select local TensorBoard writer or event-based writer for streaming
        self.writer = parts.get(self.analytic_sender_id)  # user configured config_fed_client.json for streaming
        if not self.writer:  # use local TensorBoard writer only
            self.writer = SummaryWriter(self.app_root)

        # set the training-related parameters
        # can be replaced by a config-style block
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = TorchVisionFCModel(
            model_name=self.model_name,
            num_classes=self.num_class,
            bias=True,
            pretrained=False,  # server uses pretrained weights and initializes clients
        )
        self.model = self.model.to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()
        if self.fedproxloss_mu > 0:
            self.log_info(fl_ctx, f"using FedProx loss with mu {self.fedproxloss_mu}")
            self.criterion_prox = PTFedProxLoss(mu=self.fedproxloss_mu)

        """ Data """

        # Set datalists

        train_list = load_decathlon_datalist(
            data_list_file_path=self.dataset_json,
            is_segmentation=False,
            data_list_key=self.train_set,
            base_dir=self.data_root,
        )
        val_list = load_decathlon_datalist(
            data_list_file_path=self.dataset_json,
            is_segmentation=False,
            data_list_key=self.valid_set,
            base_dir=self.data_root,
        )
        # test set is optional
        try:
            test_list = load_decathlon_datalist(
                data_list_file_path=self.dataset_json,
                is_segmentation=False,
                data_list_key=self.test_set,
                base_dir=self.data_root,
            )
        except Exception as e:
            test_list = []
            self.log_warning(fl_ctx, f"Could not create test_list: {e}")
        self.log_info(
            fl_ctx,
            f"{self.client_id}: Training Size ({self.train_set}): {len(train_list)}, "
            f"Validation Size ({self.valid_set}): {len(val_list)}, "
            f"Testing Size ({self.test_set}): {len(test_list)}",
        )

        if self.batch_size > len(train_list):
            self.batch_size = len(train_list)

        self.transform_train = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                NormalizeIntensityd(keys=["image"], subtrahend=0, divisor=255, dtype="float32"),
                RepeatChanneld(keys=["image"], repeats=3),
                NormalizeIntensityd(
                    keys=["image"],
                    subtrahend=[0.485, 0.456, 0.406],
                    divisor=[0.229, 0.224, 0.225],
                    dtype="float32",
                    channel_wise=True,
                ),
                Resized(keys=["image"], spatial_size=[224, 224]),
                ToNumpyd(keys=["label"]),
                CastToTyped(keys=["label"], dtype="long"),
                EnsureTyped(keys=["image", "label"]),
            ]
        )
        self.transform_valid = self.transform_train

        self.train_dataset = CacheDataset(
            data=train_list,
            transform=self.transform_train,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )
        self.valid_dataset = CacheDataset(
            data=val_list,
            transform=self.transform_valid,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        if test_list:  # optional
            self.test_dataset = CacheDataset(
                data=test_list,
                transform=self.transform_valid,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
            )
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.0)
        self.log_info(fl_ctx, "No-private training")

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.1)

    def finalize(self, fl_ctx: FLContext):
        # collect threads, close files here
        pass

    def local_train(
        self,
        fl_ctx,
        train_loader,
        model_global,
        abort_signal: Signal,
        val_freq: int = 0,
    ):
        for epoch in range(self.aggregation_epochs):
            if abort_signal.triggered:
                return
            self.model.train()
            epoch_len = len(train_loader)
            self.epoch_global = self.epoch_of_start_time + epoch
            avg_loss = 0.0
            for i, batch_data in enumerate(train_loader):
                if abort_signal.triggered:
                    return
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, torch.squeeze(labels, axis=1))

                # FedProx loss term
                if self.fedproxloss_mu > 0:
                    fed_prox_loss = self.criterion_prox(self.model, model_global)
                    loss += fed_prox_loss

                loss.backward()
                self.optimizer.step()
                current_step = epoch_len * self.epoch_global + i
                avg_loss += loss.item()
            avg_loss = avg_loss / len(train_loader)
            self.writer.add_scalar("train_loss", avg_loss, current_step)
            self.log_info(
                fl_ctx,
                f"Local epoch {self.client_id}: {epoch + 1}/{self.aggregation_epochs} "
                f"(lr={self.get_lr()[0]}) avg_loss: {avg_loss:.4f}",
            )
            if val_freq > 0 and epoch % val_freq == 0:
                acc, _ = self.local_valid(
                    self.valid_loader,
                    abort_signal,
                    tb_id="val_local_model",
                    fl_ctx=fl_ctx,
                )
                if acc > self.best_acc:
                    self.best_acc = acc
                    self.save_model(is_best=True)

            # update lr scheduler at end of epoch
            self.scheduler.step()

    def save_model(self, is_best=False):
        # save model
        model_weights = self.model.state_dict()
        save_model_weights = {}
        for k in model_weights.keys():
            save_model_weights[k.replace("_module.", "")] = model_weights[k]  # remove the prefix added by opacus
        save_dict = {
            "model_weights": save_model_weights,
            "epoch": self.epoch_global,
        }
        if is_best:
            save_dict.update({"best_acc": self.best_acc})
            torch.save(save_dict, self.best_local_model_file)
        else:
            torch.save(save_dict, self.local_model_file)

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get round information
        self.current_round = shareable.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(
            fl_ctx,
            f"Current/Total Round: {self.current_round + 1}/{total_rounds}",
        )
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                    n_loaded += 1
                except Exception as e:
                    raise ValueError(f"Convert weight from {var_name} failed") from e
        if n_loaded == 0:
            raise ValueError(f"No weights loaded for training! Received weight dict is {global_weights}")
        self.log_info(
            fl_ctx,
            f"Loaded global weights from {n_loaded} of {len(local_var_dict)} local layers.",
        )
        self.model.load_state_dict(local_var_dict)
        self.global_weights = local_var_dict  # update global weights so they can be accessed by inversion filter

        # local steps
        epoch_len = len(self.train_loader)
        self.log_info(fl_ctx, f"Local steps per epoch: {epoch_len}")

        # make a copy of model_global as reference for potential FedProx loss or SCAFFOLD
        model_global = copy.deepcopy(self.model)
        for param in model_global.parameters():
            param.requires_grad = False

        # local train
        self.local_train(
            fl_ctx=fl_ctx,
            train_loader=self.train_loader,
            model_global=model_global,
            abort_signal=abort_signal,
            val_freq=1,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.epoch_of_start_time += self.aggregation_epochs

        # perform valid after local train
        acc, auc = self.local_valid(
            self.valid_loader,
            abort_signal,
            tb_id="val_local_model",
            fl_ctx=fl_ctx,
        )
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)
        self.log_info(fl_ctx, f"val_local_model: accuracy: {acc:.4f}, auc: {auc:.4f}")

        # save model
        self.save_model(is_best=False)
        if acc > self.best_acc:
            self.best_acc = acc
            self.save_model(is_best=True)

        # compute delta model, global model has the primary key set
        local_weights = self.model.state_dict()
        model_diff = {}
        diff_norm = 0.0
        n_global, n_local = 0, 0
        for var_name in local_weights:
            n_local += 1
            if var_name not in global_weights:
                continue
            model_diff[var_name] = np.subtract(
                local_weights[var_name].cpu().numpy(), global_weights[var_name], dtype=np.float32
            )
            if np.any(np.isnan(model_diff[var_name])):
                self.system_panic(f"{var_name} weights became NaN...", fl_ctx)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
            n_global += 1
            diff_norm += np.linalg.norm(model_diff[var_name])
            if n_global != n_local:
                raise ValueError(
                    f"Could not compute delta for all layers! Only {n_local} local of {n_global} global layers computed..."
                )
        self.log_info(
            fl_ctx,
            f"diff norm for {n_local} local of {n_global} global layers: {diff_norm}",
        )

        # build the shareable
        dxo = DXO(data_kind=DataKind.WEIGHT_DIFF, data=model_diff)
        dxo.set_meta_prop(MetaKey.NUM_STEPS_CURRENT_ROUND, epoch_len)
        dxo.set_meta_prop(
            AppConstants.CURRENT_ROUND, self.current_round
        )  # TODO: check this is already available on server

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable.")
        return dxo.to_shareable()

    def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) -> Shareable:
        # Retrieve the best local model saved during training.
        if model_name == ModelName.BEST_MODEL:
            model_data = None
            try:
                # load model to cpu as server might or might not have a GPU
                model_data = torch.load(self.best_local_model_file, map_location="cpu")
            except Exception as e:
                raise ValueError("Unable to load best model") from e

            # Create DXO and shareable from model data.
            if model_data:
                # convert weights to numpy to support FOBS
                model_weights = model_data["model_weights"]
                for k, v in model_weights.items():
                    model_weights[k] = v.numpy()
                dxo = DXO(data_kind=DataKind.WEIGHTS, data=model_weights)
                return dxo.to_shareable()
            else:
                # Set return code.
                self.log_error(
                    fl_ctx,
                    f"best local model not found at {self.best_local_model_file}.",
                )
                return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
        else:
            raise ValueError(f"Unknown model_type: {model_name}")  # Raised errors are caught in LearnerExecutor class.

    def local_valid(self, valid_loader, abort_signal: Signal, tb_id=None, fl_ctx=None):
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            _all_pred_labels = []
            _all_labels = []
            for _i, batch_data in enumerate(valid_loader):
                if abort_signal.triggered:
                    return None
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                outputs = self.model(inputs)
                _, pred_label = torch.max(outputs.data, 1)

                total += inputs.data.size()[0]
                correct += (pred_label == torch.squeeze(labels.data, axis=1)).sum().item()
                _all_pred_labels.extend(pred_label.cpu().numpy())
                _all_labels.extend(labels.data.cpu().numpy())
            acc_metric = correct / float(total)
            if len(np.unique(_all_labels)) == 2:
                auc_metric = roc_auc_score(_all_labels, _all_pred_labels)
            else:
                auc_metric = None
            if tb_id:
                self.writer.add_scalar(tb_id + "_acc", acc_metric, self.epoch_global)
                self.writer.add_scalar(tb_id + "_auc", auc_metric, self.epoch_global)
        return acc_metric, auc_metric

    def validate(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # get validation information
        validate_type = shareable.get_header(AppConstants.VALIDATE_TYPE)
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")
        model_owner = shareable.get(ReservedHeaderKey.HEADERS).get(AppConstants.MODEL_OWNER)
        if model_owner:
            self.log_info(
                fl_ctx,
                f"Evaluating model from {model_owner} on {fl_ctx.get_identity_name()}",
            )
        else:
            model_owner = "global_model"  # evaluating global model during training

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = torch.as_tensor(global_weights[var_name], device=self.device)
                try:
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(torch.reshape(weights, local_var_dict[var_name].shape))
                    n_loaded += 1
                except Exception as e:
                    raise ValueError(f"Convert weight from {var_name} failed for {validate_type}") from e
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError(
                f"No weights loaded for validation for {validate_type}! Received weight dict is {global_weights}"
            )

        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            # perform valid before local train
            global_acc, global_auc = self.local_valid(
                self.valid_loader,
                abort_signal,
                tb_id="val_global_model",
                fl_ctx=fl_ctx,
            )
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            self.log_info(
                fl_ctx,
                f"global_model ({model_owner}): accuracy: {global_acc:.4f}, auc: {global_auc:.4f}",
            )

            return DXO(
                data_kind=DataKind.METRICS,
                data={MetaKey.INITIAL_METRICS: global_acc},
                meta={},
            ).to_shareable()

        elif validate_type == ValidateType.MODEL_VALIDATE:
            if self.test_loader is None:
                self.log_warning(fl_ctx, "No test data available. Skipping validation.")
                val_results = {"info": "Not validating on this client!"}
            else:
                # perform valid
                train_acc, train_auc = self.local_valid(self.train_loader, abort_signal)
                self.log_info(
                    fl_ctx,
                    f"training acc ({model_owner}): {train_acc:.4f}, auc: {train_auc:.4f}",
                )
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                val_acc, val_auc = self.local_valid(self.valid_loader, abort_signal)
                self.log_info(
                    fl_ctx,
                    f"validation acc ({model_owner}): {val_acc:.4f}, auc: {val_auc:.4f}",
                )
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                test_acc, test_auc = self.local_valid(self.test_loader, abort_signal)
                self.log_info(
                    fl_ctx,
                    f"testing acc ({model_owner}): {test_acc:.4f}, auc: {test_auc:.4f}",
                )
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(fl_ctx, "Evaluation finished. Returning shareable.")

                val_results = {
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                    "train_auc": train_auc,
                    "val_auc": val_auc,
                    "test_auc": test_auc,
                }
            metric_dxo = DXO(data_kind=DataKind.METRICS, data=val_results)
            return metric_dxo.to_shareable()

        else:
            return make_reply(ReturnCode.VALIDATE_TYPE_UNKNOWN)

    def get_lr(self):
        """
        This function is used to get the learning rates of the optimizer.
        """
        return [group["lr"] for group in self.optimizer.state_dict()["param_groups"]]
