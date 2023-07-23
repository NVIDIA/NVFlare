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
from timeit import default_timer as timer

import numpy as np
import torch
import torch.optim as optim
from oneshotVFL.vfl_oneshot_workflow import OSVFLDataKind, OSVFLNNConstants
from sklearn.cluster import KMeans

# from oneshotVFL.cifar10_splitnn_dataset import CIFAR10SplitNN
from splitnn.cifar10_splitnn_dataset import CIFAR10SplitNN
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from nvflare.apis.dxo import DXO, from_shareable
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.decomposers import TensorDecomposer
from nvflare.fuel.f3.stats_pool import StatsPoolManager
from nvflare.fuel.utils import fobs


class CIFAR10LearnerOneshotVFL(Learner):
    def __init__(
        self,
        dataset_root: str = "./dataset",
        intersection_file: str = None,
        lr: float = 1e-2,
        model: dict = None,
        analytic_sender_id: str = "analytic_sender",
        fp16: bool = True,
        val_freq: int = 1000,
    ):
        """Simple CIFAR-10 Trainer for split learning.

        Args:
            dataset_root: directory with CIFAR-10 data.
            intersection_file: Optional. intersection file specifying overlapping indices between both clients.
                Defaults to `None`, i.e. the whole training dataset is used.
            lr: learning rate.
            model: Split learning model.
            analytic_sender_id: id of `AnalyticsSender` if configured as a client component.
                If configured, TensorBoard events will be fired. Defaults to "analytic_sender".
            fp16: If `True`, convert activations and gradients send between clients to `torch.float16`.
                Reduces bandwidth needed for communication but might impact model accuracy.
            val_freq: how often to perform validation in rounds. Defaults to 1000. No validation if <= 0.
        """
        super().__init__()
        self.dataset_root = dataset_root
        self.intersection_file = intersection_file
        self.lr = lr
        self.model = model
        self.analytic_sender_id = analytic_sender_id
        self.fp16 = fp16
        self.val_freq = val_freq

        self.target_names = None
        self.app_root = None
        self.current_round = None
        self.num_rounds = None
        self.batch_size = None
        self.writer = None
        self.client_name = None
        self.other_client = None
        self.device = None
        self.optimizer = None
        self.criterion = None
        self.transform_train = None
        self.transform_valid = None
        self.train_dataset = None
        self.valid_dataset = None
        self.split_id = None
        self.train_activations = None
        self.train_batch_indices = None
        self.train_size = 0
        self.val_loss = []
        self.val_labels = []
        self.val_pred_labels = []
        self.compute_stats_pool = None

        self.encoder_epoch = 0
        self.clf_epoch = 0

        # use FOBS serializing/deserializing PyTorch tensors
        fobs.register(TensorDecomposer)

    def initialize(self, parts: dict, fl_ctx: FLContext):
        t_start = timer()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.encoder_epoch = 200
        self.clf_epoch = 50

        self.transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                ),
            ]
        )
        self.transform_valid = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                ),
            ]
        )

        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        self.client_name = fl_ctx.get_identity_name()
        self.split_id = self.model.get_split_id()
        self.log_info(fl_ctx, f"Running `split_id` {self.split_id} on site `{self.client_name}`")

        if self.split_id == 0:  # data side
            data_returns = "image"
        elif self.split_id == 1:  # label side
            data_returns = "label"
        else:
            raise ValueError(f"Expected split_id to be '0' or '1' but was {self.split_id}")

        if self.intersection_file is not None:
            _intersect_indices = np.loadtxt(self.intersection_file)
        else:
            _intersect_indices = None
        self.train_dataset = CIFAR10SplitNN(
            root=self.dataset_root,
            train=True,
            download=True,
            transform=self.transform_train,
            returns=data_returns,
            intersect_idx=_intersect_indices,
        )

        self.valid_dataset = CIFAR10SplitNN(
            root=self.dataset_root,
            train=False,
            download=False,
            transform=self.transform_valid,
            returns=data_returns,
            intersect_idx=None,  # TODO: support validation intersect indices
        )

        self.train_size = len(self.train_dataset)
        if self.train_size <= 0:
            raise ValueError(f"Expected train dataset size to be larger zero but got {self.train_size}")
        self.log_info(fl_ctx, f"Training with {self.train_size} overlapping indices of {self.train_dataset.orig_size}.")

        # Select local TensorBoard writer or event-based writer for streaming
        if self.split_id == 1:  # metrics can only be computed for client with labels
            self.writer = parts.get(self.analytic_sender_id)  # user configured config_fed_client.json for streaming
            if not self.writer:  # use local TensorBoard writer only
                self.writer = SummaryWriter(self.app_root)

        # register aux message handlers
        engine = fl_ctx.get_engine()

        if self.split_id == 1:
            engine.register_aux_message_handler(
                topic=OSVFLNNConstants.TASK_CALCULATE_GRADIENTS, message_handle_func=self._osvfl_calculate_gradients
            )
            engine.register_aux_message_handler(
                topic=OSVFLNNConstants.TASK_TRAIN_LABEL_SIDE, message_handle_func=self._classifier_train_label_side
            )
            engine.register_aux_message_handler(
                topic=OSVFLNNConstants.TASK_VALID, message_handle_func=self._valid_label_side
            )
            self.log_debug(fl_ctx, f"Registered aux message handlers for split_id {self.split_id}")

        self.compute_stats_pool = StatsPoolManager.add_time_hist_pool(
            "Compute_Time", "Compute time in secs", scope=self.client_name
        )

        self.compute_stats_pool.record_value(category="initialize", value=timer() - t_start)

    """ training steps """

    def _extract_features(self):
        self.model.train()
        features = []
        for _, (inputs, _) in enumerate(self.train_dataloader_no_shuffle):
            inputs = inputs.to(self.device)
            features.append(self.model(inputs))

        features = torch.cat(features, dim=0)
        features = features.view(features.size(0), -1)
        return features.detach().requires_grad_()

    def _osvfl_calculate_gradients(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        """train aux message handler"""
        t_start = timer()
        if self.split_id != 1:
            raise ValueError(
                f"Expected `split_id` 1. It doesn't make sense to run `_aux_train_label_side` with `split_id` {self.split_id}"
            )

        self.current_round = request.get_header(AppConstants.CURRENT_ROUND)
        self.num_rounds = request.get_header(AppConstants.NUM_ROUNDS)
        self.log_debug(fl_ctx, f"Calculate gradients in round {self.current_round} of {self.num_rounds} rounds.")

        dxo = from_shareable(request)
        if dxo.data_kind != OSVFLDataKind.FEATURES:
            raise ValueError(f"Expected data kind {OSVFLDataKind.FEATURES} but received {dxo.data_kind}")

        features = dxo.data.get(OSVFLNNConstants.DATA)

        if features is None:
            raise ValueError("No features in DXO!")

        features = fobs.loads(features)
        print(features.shape)

        feature_dataset = copy.deepcopy(self.train_dataset)
        if self.fp16:
            features = features.type(torch.float32)  # return to default pytorch precision
        feature_dataset.data = features
        feature_dataset.transform = None
        feature_dataloader = torch.utils.data.DataLoader(feature_dataset, batch_size=self.batch_size, shuffle=False)

        gradient = []

        self.model.eval()
        for _, (activations, labels) in enumerate(feature_dataloader):
            activations, labels = activations.to(self.device), labels.to(self.device)
            activations.requires_grad_(True)

            self.optimizer.zero_grad()
            pred = self.model.forward(activations)
            loss = self.criterion(pred, labels)
            loss.backward()

            if not isinstance(activations.grad, torch.Tensor):
                raise ValueError("No valid gradients available!")
            # gradient to be returned to other client
            if self.fp16:
                gradient.append(activations.grad.type(torch.float16))
            else:
                gradient.append(activations.grad)

        gradient = torch.cat(gradient).cpu().numpy()

        self.log_debug(fl_ctx, "_osvfl_calculate_gradients finished.")
        return_shareable = DXO(
            data={OSVFLNNConstants.DATA: fobs.dumps(gradient)}, data_kind=OSVFLDataKind.GRADIENTS
        ).to_shareable()

        self.compute_stats_pool.record_value(category="_osvfl_calculate_gradients", value=timer() - t_start)

        self.log_debug(fl_ctx, f"Sending partial gradients return_shareable: {type(return_shareable)}")
        return return_shareable

    def _cluster_gradients(self, gradients, fl_ctx):
        num_clusters = 10
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=1).fit(gradients)
        cluster_labels = kmeans.labels_
        self.log_info(fl_ctx, "_cluster_gradients finished.")
        return cluster_labels

    def _local_train(self, cluster_labels, fl_ctx):
        l_labels = torch.LongTensor(cluster_labels)
        local_train_dataset = copy.deepcopy(self.train_dataset)
        local_train_dataset.target = l_labels
        local_train_datloader = torch.utils.data.DataLoader(
            local_train_dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model.train()
        for e in range(self.encoder_epoch):
            loss_ep = []
            for _, (inputs, labels) in enumerate(local_train_datloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model.forward_complt(inputs)
                loss = self.criterion(pred, labels)
                loss.backward()

                self.optimizer.step()

                loss_ep.append(loss.item())

            loss_epoch = sum(loss_ep) / len(loss_ep)

            if e % 10 == 0:
                self.model.eval()
                correct = 0
                for _, (inputs, labels) in enumerate(local_train_datloader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    pred = self.model.forward_complt(inputs)
                    _, pred_labels = torch.max(pred, 1)
                    correct += (pred_labels == labels).sum()
                acc = correct / len(cluster_labels)

                self.log_info(
                    fl_ctx,
                    f"Epoch {e}/{self.encoder_epoch} train_loss: {loss_epoch:.4f}, train_accuracy: {acc:.4f}",
                )

    def _classifier_train_label_side(self, topic: str, request: Shareable, fl_ctx: FLContext):
        t_start = timer()
        if self.split_id != 1:
            raise ValueError(
                f"Expected `split_id` 1. It doesn't make sense to run `_aux_train_label_side` with `split_id` {self.split_id}"
            )

        self.current_round = request.get_header(AppConstants.CURRENT_ROUND)
        self.num_rounds = request.get_header(AppConstants.NUM_ROUNDS)
        self.log_debug(fl_ctx, f"Calculate gradients in round {self.current_round} of {self.num_rounds} rounds.")

        dxo = from_shareable(request)
        if dxo.data_kind != OSVFLDataKind.FEATURES:
            raise ValueError(f"Expected data kind {OSVFLDataKind.FEATURES} but received {dxo.data_kind}")

        features = dxo.data.get(OSVFLNNConstants.DATA)
        if features is None:
            raise ValueError("No features in DXO!")
        features = fobs.loads(features)

        feature_dataset = copy.deepcopy(self.train_dataset)
        if self.fp16:
            features = features.type(torch.float32)  # return to default pytorch precision
        feature_dataset.data = features
        feature_dataset.transform = None
        feature_dataloader = torch.utils.data.DataLoader(feature_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.train()
        for e in range(self.encoder_epoch):
            loss_ep = []
            for _, (inputs, labels) in enumerate(feature_dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model.forward(inputs)
                loss = self.criterion(pred, labels)
                loss.backward()

                self.optimizer.step()

                loss_ep.append(loss.item())

            loss_epoch = sum(loss_ep) / len(loss_ep)

            if e % 10 == 0:
                self.model.eval()
                correct = 0
                for _, (inputs, labels) in enumerate(feature_dataloader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    pred = self.model.forward(inputs)
                    _, pred_labels = torch.max(pred, 1)
                    correct += (pred_labels == labels).sum()
                acc = correct / features.shape[0]

                self.log_info(
                    fl_ctx,
                    f"Label Side Epoch {e}/{self.encoder_epoch} train_loss: {loss_epoch:.4f}, train_accuracy: {acc:.4f}",
                )

        return make_reply(ReturnCode.OK)

    def _valid_label_side(self, topic: str, request: Shareable, fl_ctx: FLContext):
        t_start = timer()
        if self.split_id != 1:
            raise ValueError(
                f"Expected `split_id` 1. It doesn't make sense to run `_aux_train_label_side` with `split_id` {self.split_id}"
            )

        dxo = from_shareable(request)
        if dxo.data_kind != OSVFLDataKind.FEATURES:
            raise ValueError(f"Expected data kind {OSVFLDataKind.FEATURES} but received {dxo.data_kind}")

        features = dxo.data.get(OSVFLNNConstants.DATA)
        if features is None:
            raise ValueError("No features in DXO!")
        features = fobs.loads(features)

        feature_dataset = copy.deepcopy(self.valid_dataset)
        if self.fp16:
            features = features.type(torch.float32)  # return to default pytorch precision
        feature_dataset.data = features
        feature_dataset.transform = None
        feature_dataloader = torch.utils.data.DataLoader(feature_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        correct = 0
        for _, (inputs, labels) in enumerate(feature_dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            pred = self.model.forward(inputs)
            _, pred_labels = torch.max(pred, 1)
            correct += (pred_labels == labels).sum()
        acc = correct / features.shape[0]

        self.log_info(
            fl_ctx,
            f"Label Side test_accuracy: {acc:.4f}",
        )

        return make_reply(ReturnCode.OK)

    # Model initialization task (one time only in beginning)
    def init_model(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        t_start = timer()
        # Check abort signal
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # update local model weights with received weights
        dxo = from_shareable(shareable)
        global_weights = dxo.data

        # Before loading weights, tensors might need to be reshaped to support HE for secure aggregation.
        local_var_dict = self.model.state_dict()
        model_keys = global_weights.keys()
        n_loaded = 0
        for var_name in local_var_dict:
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            if var_name in model_keys:
                weights = global_weights[var_name]
                try:
                    # reshape global weights to compute difference later on
                    global_weights[var_name] = np.reshape(weights, local_var_dict[var_name].shape)
                    # update the local dict
                    local_var_dict[var_name] = torch.as_tensor(global_weights[var_name])
                    n_loaded += 1
                except Exception as e:
                    raise ValueError(f"Convert weight from {var_name} failed.") from e
        self.model.load_state_dict(local_var_dict)
        if n_loaded == 0:
            raise ValueError("No global weights loaded!")

        self.compute_stats_pool.record_value(category="init_model", value=timer() - t_start)

        self.log_info(fl_ctx, "init_model finished.")

        return make_reply(ReturnCode.OK)

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        t_start = timer()
        """main training logic"""
        engine = fl_ctx.get_engine()

        self.num_rounds = shareable.get_header(AppConstants.NUM_ROUNDS)
        if not self.num_rounds:
            raise ValueError("No number of rounds available.")
        self.batch_size = shareable.get_header(OSVFLNNConstants.BATCH_SIZE)
        self.target_names = np.asarray(
            shareable.get_header(OSVFLNNConstants.TARGET_NAMES)
        )  # convert to array for string matching below
        self.other_client = self.target_names[self.target_names != self.client_name][0]
        self.log_info(fl_ctx, f"Starting training of {self.num_rounds} rounds with batch size {self.batch_size}")

        gradients = None  # initial gradients

        if self.split_id != 0:
            self.compute_stats_pool.record_value(category="train", value=timer() - t_start)
            return make_reply(ReturnCode.OK)  # only run this logic on first site

        self.train_dataloader_shuffle = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

        self.train_dataloader_no_shuffle = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=False
        )

        self.valid_dataloader = torch.utils.data.DataLoader(
            self.valid_dataset, batch_size=self.batch_size, shuffle=False
        )

        for _curr_round in range(self.num_rounds):
            self.current_round = _curr_round
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            self.log_debug(fl_ctx, f"Starting current round={self.current_round} of {self.num_rounds}.")
            # self.train_batch_indices = np.random.randint(0, self.train_size - 1, self.batch_size)

            # first round: site-1 extracts features and send to site-2; site-2 return gradients back to site-1
            if _curr_round == 0:
                fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self.current_round, private=True, sticky=False)

                # site-1 extract features
                features = self._extract_features()

                # package features
                dxo = DXO(data={OSVFLNNConstants.DATA: fobs.dumps(features)}, data_kind=OSVFLDataKind.FEATURES)
                data_shareable = dxo.to_shareable()

                # add meta data for transmission
                data_shareable.set_header(AppConstants.CURRENT_ROUND, self.current_round)
                data_shareable.set_header(AppConstants.NUM_ROUNDS, self.num_rounds)
                data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, self.current_round)

                # send clustering request to site-2
                result = engine.send_aux_request(
                    targets=self.other_client,
                    topic=OSVFLNNConstants.TASK_CALCULATE_GRADIENTS,
                    request=data_shareable,
                    timeout=OSVFLNNConstants.TIMEOUT,
                    fl_ctx=fl_ctx,
                )

                # check returned results (gradients)
                shareable = result.get(self.other_client)
                if shareable is not None:
                    dxo = from_shareable(shareable)
                    if dxo.data_kind != OSVFLDataKind.GRADIENTS:
                        raise ValueError(f"Expected data kind {OSVFLDataKind.GRADIENTS} but received {dxo.data_kind}")
                    gradients = dxo.data.get(OSVFLNNConstants.DATA)
                    gradients = fobs.loads(gradients)
                else:
                    raise ValueError(f"No message returned from {self.other_client}!")

            # second round: site-1 conducts clustering, local training, and sending features to site-2;
            # site-2 trains the classifier
            elif _curr_round == 1:
                # site-1 conducts clustering and local training
                cluster_labels = self._cluster_gradients(gradients, fl_ctx)
                self._local_train(cluster_labels, fl_ctx)
                features = self._extract_features()

                # site-1 packages features
                dxo = DXO(data={OSVFLNNConstants.DATA: fobs.dumps(features)}, data_kind=OSVFLDataKind.FEATURES)
                data_shareable = dxo.to_shareable()
                data_shareable.set_header(AppConstants.CURRENT_ROUND, self.current_round)
                data_shareable.set_header(AppConstants.NUM_ROUNDS, self.num_rounds)
                data_shareable.add_cookie(AppConstants.CONTRIBUTION_ROUND, self.current_round)

                # site-1 sends features to site-2; site-2 trains the classifier
                engine.send_aux_request(
                    targets=self.other_client,
                    topic=OSVFLNNConstants.TASK_TRAIN_LABEL_SIDE,
                    request=data_shareable,
                    timeout=OSVFLNNConstants.TIMEOUT,
                    fl_ctx=fl_ctx,
                )
                try:
                    self._validate(fl_ctx)
                except Exception as e:
                    self.log_info(fl_ctx, "Valiate exit with exception {}".format(e))
                    return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            self.log_debug(fl_ctx, f"Ending current round={self.current_round}.")

            # if self.val_freq > 0:
            #     if _curr_round % self.val_freq == 0:
            #         self._validate(fl_ctx)

        self.compute_stats_pool.record_value(category="train", value=timer() - t_start)

        return make_reply(ReturnCode.OK)

    def _validate(self, fl_ctx: FLContext):
        t_start = timer()
        engine = fl_ctx.get_engine()

        self.model.eval()
        features = []
        for _, (inputs, _) in enumerate(self.valid_dataloader):
            inputs = inputs.to(self.device)
            features.append(self.model(inputs))

        features = torch.cat(features, dim=0)
        features = features.view(features.size(0), -1)

        dxo = DXO(data={OSVFLNNConstants.DATA: fobs.dumps(features)}, data_kind=OSVFLDataKind.FEATURES)

        data_shareable = dxo.to_shareable()

        # send to other side to validate
        engine.send_aux_request(
            targets=self.other_client,
            topic=OSVFLNNConstants.TASK_VALID,
            request=data_shareable,
            timeout=OSVFLNNConstants.TIMEOUT,
            fl_ctx=fl_ctx,
        )

        self.compute_stats_pool.record_value(category="_validate", value=timer() - t_start)

        self.log_debug(fl_ctx, "finished validation.")
