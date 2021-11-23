from nvflare.apis.dxo import DataKind, from_bytes
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.formatter import Formatter
from nvflare.app_common.app_constant import AppConstants


class NPFormatter(Formatter):
    def __init__(self) -> None:
        super().__init__()

    def format(self, fl_ctx: FLContext) -> str:
        """The format function gets validation shareable locations from the dictionary. It loads each shareable,
        get the validation results and converts it into human readable string.

        Args:
            fl_ctx (FLContext): FLContext object.

        Returns:
            str: Human readable validation results.
        """
        # Get the val shareables
        validation_shareables_dict = fl_ctx.get_prop(AppConstants.VALIDATION_RESULT, {})

        # Result dictionary
        res = {}

        try:
            # This is a 2d dictionary with each validation result at
            # validation_shareables_dict[data_client][model_client]
            for data_client in validation_shareables_dict.keys():
                validation_dict = validation_shareables_dict[data_client]
                if validation_dict:
                    res[data_client] = {}
                    for model_name in validation_dict.keys():
                        dxo_path = validation_dict[model_name]

                        # Load the shareable
                        with open(dxo_path, "rb") as f:
                            metric_dxo = from_bytes(f.read())

                        # Get metrics from shareable
                        if metric_dxo and metric_dxo.data_kind == DataKind.METRICS:
                            metrics = metric_dxo.data
                            res[data_client][model_name] = metrics
        except Exception as e:
            self.log_error(fl_ctx, f"Exception: {e.__str__()}")

        return f"{res}"
