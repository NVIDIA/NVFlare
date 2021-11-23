import os
from test.app_testing.app_result_validator import AppResultValidator

import numpy as np


def check_sag_results(server_data, client_data, run_data):
    run_number = run_data["run_number"]
    server_dir = server_data["server_path"]

    server_run_dir = os.path.join(server_dir, "run_" + str(run_number))
    intended_model = np.array(
        [[4., 5., 6.],
         [7., 8., 9.],
         [10., 11., 12.]],
        dtype="float32",
    )

    if not os.path.exists(server_run_dir):
        print(
            f"check_sag_results: server run dir {server_run_dir} doesn't exist."
        )
        return False

    models_dir = os.path.join(server_run_dir, "models")
    if not os.path.exists(models_dir):
        print(f"check_sag_results: models dir {models_dir} doesn't exist.")
        return False

    model_path = os.path.join(models_dir, "server.npy")
    if not os.path.isfile(model_path):
        print(f"check_sag_results: model_path {model_path} doesn't exist.")
        return False

    try:
        data = np.load(model_path)
        print(f"check_sag_result: Data loaded: {data}.")
        np.testing.assert_equal(data, intended_model)
    except Exception as e:
        print(f"Exception in validating ScatterAndGather model: {e.__str__()}")
        return False

    return True


class SAGResultValidator(AppResultValidator):
    def __init__(self):
        super(SAGResultValidator, self).__init__()

    def validate_results(self, server_data, client_data, run_data) -> bool:
        # pass
        # print(f"CrossValValidator server data: {server_data}")
        # print(f"CrossValValidator client data: {client_data}")
        # print(f"CrossValValidator run data: {run_data}")

        fed_avg_result = check_sag_results(server_data, client_data, run_data)

        print(f"ScatterAndGather Result: {fed_avg_result}")

        if not fed_avg_result:
            raise ValueError("Scatter and gather failed.")

        return fed_avg_result
