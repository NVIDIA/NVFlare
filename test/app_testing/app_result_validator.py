import logging


class AppResultValidator(object):

    def __init__(self):
        super(AppResultValidator, self).__init__()
        self.logger = logging.getLogger("AppValidator")

    def validate_results(self, server_data, client_data, run_data) -> bool:
        pass

