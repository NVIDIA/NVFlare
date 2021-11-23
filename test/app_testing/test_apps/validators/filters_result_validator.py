import os
from test.app_testing.app_result_validator import AppResultValidator

import numpy as np


class FiltersResultValidator(AppResultValidator):
    def validate_results(self, server_data, client_data, run_data) -> bool:
        run_number = run_data["run_number"]
        server_dir = server_data["server_path"]

        server_run_dir = os.path.join(server_dir, f"run_{run_number}")

        if not os.path.exists(server_run_dir):
            print(
                f"FiltersResultValidator: server run dir {server_run_dir} doesn't exist."
            )
            return False

        models_dir = os.path.join(server_run_dir, "models")
        if not os.path.exists(models_dir):
            print(f"FiltersResultValidator: models dir {models_dir} doesn't exist.")
            return False

        model_path = os.path.join(models_dir, "server.npy")
        if not os.path.isfile(model_path):
            print(f"FiltersResultValidator: model_path {model_path} doesn't exist.")
            return False

        try:
            data = np.load(model_path)
            print(f"FiltersResultValidator: Data loaded: {data}.")
        except Exception as e:
            print(f"Exception in FiltersResultValidator: {e.__str__()}")
            return False

        return True
