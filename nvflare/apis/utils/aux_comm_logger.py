import logging
import threading

from nvflare.app_common.widgets.streaming import LogSender


class AuxCommLogger:
    def __init__(self, logger):
        """
        LogSender sends the logging data to the server. It must remove itself when calling the logger function
        during the process.
        Args:
            logger: logger object
        """
        self.logger = logger
        self.lock = threading.Lock()

    def get_non_logsender_handlers(self):
        for handler in logging.root.handlers:
            if not isinstance(handler, LogSender):
                self.logger.addHandler(handler)
        self.org_propagate = self.logger.propagate
        self.logger.propagate = False

    def remove_root_handlers(self):
        for handler in logging.root.handlers:
            if not isinstance(handler, LogSender):
                self.logger.removeHandler(handler)
        self.logger.propagate = self.org_propagate

    def debug(self, msg, *args, **kwargs):
        with self.lock:
            self.get_non_logsender_handlers()
            self.logger.debug(msg, *args, **kwargs)
            self.remove_root_handlers()

    def info(self, msg, *args, **kwargs):
        with self.lock:
            self.get_non_logsender_handlers()
            self.logger.info(msg, *args, **kwargs)
            self.remove_root_handlers()

    def warn(self, msg, *args, **kwargs):
        with self.lock:
            self.get_non_logsender_handlers()
            self.logger.warn(msg, *args, **kwargs)
            self.remove_root_handlers()

    def error(self, msg, *args, **kwargs):
        with self.lock:
            self.get_non_logsender_handlers()
            self.logger.error(msg, *args, **kwargs)
            self.remove_root_handlers()

    def exception(self, msg, *args, **kwargs):
        with self.lock:
            self.get_non_logsender_handlers()
            self.logger.exception(msg, *args, **kwargs)
            self.remove_root_handlers()

    def critical(self, msg, *args, **kwargs):
        with self.lock:
            self.get_non_logsender_handlers()
            self.logger.critical(msg, *args, **kwargs)
            self.remove_root_handlers()

