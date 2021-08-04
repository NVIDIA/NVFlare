import numpy as np

from nvflare.apis.filter import Filter
from nvflare.apis.fl_constant import FLConstants, ShareableKey, ShareableValue
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class ExcludeVars(Filter):
    """
        Exclude/Remove variables from Sharable

    Args:
        exclude_vars: if not specified (None), all layers are being encrypted;
                      if list of variable/layer names, only specified variables are excluded;
    """

    def __init__(self, exclude_vars=None):
        super().__init__()
        self.exclude_vars = exclude_vars

        if self.exclude_vars is not None:
            assert isinstance(self.exclude_vars, list), (
                "Provide a list of layer names "
            )
            for var in self.exclude_vars:
                assert isinstance(var, str), (
                    "encrypt_layers needs to be a list of layer names to encrypt."
                )
            self.logger.info(f"Excluding {self.exclude_vars} from shareable")
        else:
            self.logger.info(f"Not excluding anything")

    def process(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        weights = shareable[ShareableKey.MODEL_WEIGHTS]

        # set excluded variables to all zeros
        n_excluded = 0
        var_names = list(weights.keys()) 
        n_vars = len(var_names)
        for var_name in var_names:
            if self.exclude_vars:
                if var_name in self.exclude_vars:
                    self.logger.info(f"Excluding {var_name}")
                    weights[var_name] = np.zeros(weights[var_name].shape)        
                    n_excluded += 1
        self.logger.info(f"Excluded {n_excluded} of {n_vars} variables")

        shareable[ShareableKey.MODEL_WEIGHTS] = weights
        return shareable
