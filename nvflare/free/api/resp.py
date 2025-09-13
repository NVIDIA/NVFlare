import time


class Resp:

    def __init__(self, process_cb, cb_kwargs):
        self.result = None
        self.exception = None
        self.resp_time = None
        self.process_cb = process_cb
        self.cb_kwargs = cb_kwargs

    def set_result(self, result):
        if self.process_cb:
            result = self.process_cb(result, **self.cb_kwargs)
        self.result = result
        self.resp_time = time.time()

    def set_exception(self, ex):
        self.exception = ex
        self.resp_time = time.time()
