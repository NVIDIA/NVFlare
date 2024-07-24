from typing import Dict, Optional

from nvflare.app_common.abstract.fl_model import FLModel


class GenericTask(FLModel):
    def __init__(self, meta: Optional[Dict] = None):
        super().__init__(
            params_type=None,
            params={},
            optimizer_params=None,
            metrics=None,
            start_round=0,
            current_round=0,
            total_rounds=1,
            meta=meta,
        )
