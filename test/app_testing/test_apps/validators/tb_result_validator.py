import os
from test.app_testing.app_result_validator import AppResultValidator

TB_PATH = "tb_events"


class TBResultValidator(AppResultValidator):
    def validate_results(self, server_data, client_data, run_data) -> bool:
        run_number = run_data["run_number"]
        server_path = server_data["server_path"]

        server_run_dir = os.path.join(server_path, f"run_{run_number}")
        server_tb_root_dir = os.path.join(server_run_dir, TB_PATH)
        if not os.path.exists(server_tb_root_dir):
            return False

        for i, client_path in enumerate(client_data["client_paths"]):
            client_run_dir = os.path.join(client_path, f"run_{run_number}")
            client_side_client_tb_dir = os.path.join(client_run_dir, TB_PATH, client_data["client_names"][i])
            if not os.path.exists(client_side_client_tb_dir):
                return False

            server_side_client_tb_dir = os.path.join(server_tb_root_dir, client_data["client_names"][i])
            if not os.path.exists(server_side_client_tb_dir):
                return False
        return True
