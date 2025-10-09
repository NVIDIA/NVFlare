import torch

from nvflare.apis.fl_context import FLContext
from nvflare.app_opt.tensor_stream.types import TensorParams
from nvflare.app_opt.tensor_stream.utils import get_dxo_from_ctx
from nvflare.fuel.utils.log_utils import get_obj_logger


class TensorStore:

    def __init__(self, task_id: str, task_name: str, ctx_prop_key: str):
        self.task_id = task_id
        self.task_name = task_name
        self.ctx_prop_key = ctx_prop_key
        self.tensors_map: dict[str, TensorParams] = {}
        self.logger = get_obj_logger(self)

    def parse(self, fl_ctx: FLContext):
        dxo = get_dxo_from_ctx(fl_ctx, self.ctx_prop_key, [self.task_name])
        for key, value in dxo.data.items():
            # auto-detect tensor stored on root keys
            if not isinstance(value, dict) and "" not in self.tensors_map:
                self.tensors_map[""] = dxo.data
                break
            elif isinstance(value, dict) and key not in self.tensors_map:
                self.tensors_map[key] = value

    def get(self) -> dict[str, TensorParams]:
        return self.tensors_map

    def clear(self):
        self.tensors_map.clear()
