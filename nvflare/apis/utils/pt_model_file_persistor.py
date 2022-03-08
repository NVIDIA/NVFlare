import os
import re
from collections import OrderedDict

import torch
from dlmed.utils.wfconf import Configurator

from flare.apis.event_type import EventType
from flare.apis.fl_constant import CrossValConstants, FLConstants
from flare.apis.fl_context import FLContext
from flare.apis.learnable import Learnable
from flare.apis.model_persistor import ModelPersistor
from flare.utils.ml_model_registry import MLModelEntry


class PTFileModelPersistor(ModelPersistor):
    FL_PACKAGES = ["flare"]
    FL_MODULES = ["server", "client", "components", "handlers", "pt", "app"]

    def __init__(self, exclude_vars=None, model=None):
        super().__init__()
        self.exclude_vars = re.compile(exclude_vars) if exclude_vars else None
        self.model_config = model
        self.log_dir = None
        self.ckpt_preload_path = None
        self.train_conf = None

    def _initialize(self, fl_ctx: FLContext):
        mmar_root = fl_ctx.get_prop(FLConstants.TRAIN_ROOT)
        config_file = os.path.join(mmar_root, fl_ctx.get_prop(FLConstants.ARGS).train_config)

        conf = Configurator(
            mmar_root=mmar_root,
            cmd_vars={},
            env_config={},
            wf_config_file_name="/tmp/fl_server/config_train.json",
            base_pkgs=PTFileModelPersistor.FL_PACKAGES,
            module_names=PTFileModelPersistor.FL_MODULES,
        )

        self.model = conf.build_component(self.model_config)

    def load_model(self, fl_ctx: FLContext):
        """Convert initialised model into protobuf message.
        This function sets self.model to a ModelData protobuf message.

        Args:
            fl_ctx (FLContext): FL Context delivered by workflow

        Returns:
            Model: a model populated with storaged pointed by fl_ctx
        """

        self._initialize(fl_ctx)

        if self.ckpt_preload_path:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            data = torch.load(self.ckpt_preload_path, map_location=device)
            # "checkpoint may contain 'model', 'optimizer', 'lr_scheduler', etc. or only contain model dict directly."
            var_dict = data.get("model", None)
            if var_dict is None:
                var_dict = data
            else:
                train_conf = data.get("train_conf", None)
                if train_conf is not None:
                    self.train_conf = train_conf
        else:
            # if no pretrained model provided, use the generated network weights from MMAR config
            # note that, if set "determinism" in the config, the init model weights will always be the same
            var_dict = self.model.state_dict() if self.model is not None else OrderedDict()

        return self._to_model(var_dict)

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.BEFORE_MODEL_UPDATE and fl_ctx.get_prop(FLConstants.IS_BEST):
            print(f"Start to save the best global model to: {self._best_ckpt_save_path}")
            self.save_model_file(fl_ctx.get_model(), self._best_ckpt_save_path)
            self.register_model_for_cross_validation(fl_ctx, "best_FL_global_model", self._best_ckpt_save_path)
            fl_ctx.remove_prop(FLConstants.IS_BEST)
        elif event == EventType.START_RUN:
            print("Got EventType.START_RUN event.")
            train_root = fl_ctx.get_prop(FLConstants.TRAIN_ROOT)
            log_dir = fl_ctx.get_prop(FLConstants.LOG_DIR)
            if log_dir:
                self.log_dir = os.path.join(train_root, log_dir)
            else:
                self.log_dir = train_root

            self._ckpt_save_path = os.path.join(self.log_dir, "FL_global_model.pt")
            self._best_ckpt_save_path = os.path.join(self.log_dir, "best_FL_global_model.pt")

            ckpt_preload_path = fl_ctx.get_prop(FLConstants.CKPT_PRELOAD_PATH)
            if ckpt_preload_path:
                self.ckpt_preload_path = os.path.join(train_root, ckpt_preload_path)

            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

    def save_model_file(self, model: Learnable, save_path: str):
        weights_dict = OrderedDict()
        for var_name, nd in model.items():
            weights_dict[var_name] = torch.as_tensor(nd)

        save_dict = OrderedDict()
        save_dict["model"] = weights_dict
        if self.train_conf is not None:
            save_dict["train_conf"] = self.train_conf
        else:
            save_dict["train_conf"] = {"train": {"model": self.model_config}}

        print(f"Before the torch.save(), path: {save_path}")
        torch.save(save_dict, save_path)
        print(f"Save complete, path: {save_path}")

    def register_model_for_cross_validation(self, fl_ctx, model_name, save_path):
        model_registry = fl_ctx.get_prop(CrossValConstants.ML_MODEL_REGISTRY)
        if model_registry:
            ml_model_entry = MLModelEntry(model_name)
            ml_model_entry.add_files({"PT_MODEL": save_path})
            model_registry.register_model(model_name, ml_model_entry, override=True)

    def save_model(self, model: Learnable, fl_ctx: FLContext):
        self.save_model_file(model, self._ckpt_save_path)
        self.register_model_for_cross_validation(fl_ctx, "FL_global_model", self._ckpt_save_path)

    def finalize(self, fl_ctx: FLContext):
        pass

    def _to_model(self, var_dict):
        model = Learnable()
        for var_name in var_dict:
            if self.exclude_vars and self.exclude_vars.search(var_name):
                continue
            model.update({var_name: var_dict[var_name].cpu().numpy()})

        return model
