import os
from typing import List

import torch.cuda

from app_testing.test_apps.pt.custom.net import SimpleNetwork
from nvflare.apis.dxo import DXO
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import model_learnable_to_dxo
from nvflare.app_common.abstract.model_locator import ModelLocator
from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager
from test.app_testing.test_apps.pt.custom.constants import PTConstants


class PTModelLocator(ModelLocator):

    def __init__(self, exclude_vars=None, model=None):
        super(PTModelLocator, self).__init__()

        self.model = SimpleNetwork()
        self.exclude_vars = exclude_vars

    def get_model_names(self, fl_ctx: FLContext) -> List[str]:
        return [PTConstants.PTServerName]

    def locate_model(self, model_name, fl_ctx: FLContext) -> DXO:
        if model_name == PTConstants.PTServerName:
            server_run_dir = fl_ctx.get_engine().get_workspace().get_app_dir(fl_ctx.get_run_number())
            model_path = os.path.join(server_run_dir, "models", PTConstants.PTFileModelName)
            if not os.path.exists(model_path):
                return None

            # Load the torch model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            data = torch.load(model_path, map_location=device)

            # Setup the persistence manager.
            if self.model:
                default_train_conf = {"train": {"model": type(self.model).__name__}}
            else:
                default_train_conf = None

            # Use persistence manager to get learnable
            persistence_manager = PTModelPersistenceFormatManager(data, default_train_conf=default_train_conf)
            ml = persistence_manager.to_model_learnable(exclude_vars=None)

            # Create dxo and return
            return model_learnable_to_dxo(ml)
        else:
            self.log_exception(fl_ctx, f"PTModelLocator doesn't recognize name: {model_name}")
            return None
