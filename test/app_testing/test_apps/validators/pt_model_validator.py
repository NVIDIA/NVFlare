import os
from collections import OrderedDict

import numpy as np

from test.app_testing.app_result_validator import AppResultValidator


def check_pt_results(server_data, client_data, run_data):
    run_number = run_data["run_number"]
    server_dir = server_data["server_path"]

    server_run_dir = os.path.join(server_dir, "run_" + str(run_number))
    intended_model = OrderedDict([('model',
                                   OrderedDict([('linear.weight',
                                                np.array([[-0.1234, -0.1452, -0.1570]])),
                                                ('linear.bias', np.array([0.4643]))])),
                                  ('train_conf', {'train': {'model': 'SimpleNetwork'}})])

    if not os.path.exists(server_run_dir):
        print(
            f"check_sag_results: server run dir {server_run_dir} doesn't exist."
        )
        return False

    models_dir = os.path.join(server_run_dir, "app_server")
    if not os.path.exists(models_dir):
        print(f"check_sag_results: models dir {models_dir} doesn't exist.")
        return False

    model_path = os.path.join(models_dir, "FL_global_model.pt")
    if not os.path.isfile(model_path):
        print(f"check_sag_results: model_path {model_path} doesn't exist.")
        return False

    return True


class PTModelValidator(AppResultValidator):

    def __init__(self):
        super(PTModelValidator, self).__init__()

    def validate_results(self, server_data, client_data, run_data) -> bool:
        return check_pt_results(server_data, client_data, run_data)