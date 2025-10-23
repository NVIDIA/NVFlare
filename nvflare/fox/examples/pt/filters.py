from typing import Any

from nvflare.fox.api.ctx import Context
from nvflare.fox.api.filter import CallFilter, ResultFilter
from nvflare.fox.sys.downloader import Downloader, download_tensors
from nvflare.fuel.utils.log_utils import get_obj_logger


class OutgoingModelCallFilter(CallFilter):

    def __init__(self, model_arg_name: str):
        super().__init__()
        self.model_arg_name = model_arg_name
        self.logger = get_obj_logger(self)

    def filter_call(self, func_kwargs: dict, context: Context):
        arg_value = func_kwargs.get(self.model_arg_name)
        if not arg_value:
            return func_kwargs

        num_receivers = context.target_group_size
        self.logger.info(f"target group size={num_receivers}")

        downloader = Downloader(
            num_receivers=context.target_group_size,
            ctx=context,
            timeout=5.0,
        )
        model = downloader.add_tensors(arg_value, 0)
        func_kwargs[self.model_arg_name] = model
        return func_kwargs


class IncomingModelCallFilter(CallFilter):

    def __init__(self, model_arg_name: str):
        super().__init__()
        self.model_arg_name = model_arg_name
        self.logger = get_obj_logger(self)

    def filter_call(self, func_kwargs: dict, context: Context):
        arg_value = func_kwargs.get(self.model_arg_name)
        if not arg_value:
            return func_kwargs

        err, model = download_tensors(ref=arg_value, ctx=context, per_request_timeout=5.0)
        if err:
            self.logger.error(f"error filtering call arg {arg_value}: {err}")
        else:
            func_kwargs[self.model_arg_name] = model
            return func_kwargs
        return func_kwargs


class OutgoingModelResultFilter(ResultFilter):

    def filter_result(self, result: Any, context: Context):
        if not isinstance(result, dict):
            return result

        downloader = Downloader(
            num_receivers=1,
            ctx=context,
            timeout=5.0,
        )
        return downloader.add_tensors(result, 0)


class IncomingModelResultFilter(ResultFilter):

    def __init__(self):
        super().__init__()
        self.logger = get_obj_logger(self)

    def filter_result(self, result: Any, context: Context):
        err, model = download_tensors(ref=result, ctx=context, per_request_timeout=5.0)
        if err:
            self.logger.error(f"error filtering result {result}: {err}")
            return result
        else:
            return model
