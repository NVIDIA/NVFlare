# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import importlib
import os
import sys
import time
import traceback
from test.app_testing.admin_controller import AdminController
from test.app_testing.site_launcher import SiteLauncher

import yaml


def get_module_class_from_full_path(full_path):
    tokens = full_path.split(".")
    cls_name = tokens[-1]
    mod_name = ".".join(tokens[: len(tokens) - 1])
    return mod_name, cls_name


def read_yaml(yaml_file_path):
    if not os.path.exists(yaml_file_path):
        print(f"Yaml file doesnt' exist at {yaml_file_path}")
        return None

    with open(yaml_file_path, "rb") as f:
        data = yaml.safe_load(f)

    return [(x["app_name"], x["validators"]) for x in data["tests"]]


if __name__ == "__main__":
    site_launcher = None
    cleanup = True
    try:
        parser = argparse.ArgumentParser(description="Run NVFlare tests")
        parser.add_argument("--poc", "-p", type=str, help="Poc Directory")
        parser.add_argument("--n_clients", "-n", type=int, help="Number of clients.")
        parser.add_argument("--app_path", "-ap", type=str, help="Directory for the apps.")
        parser.add_argument("--yaml", "-y", type=str, help="Yaml test config path.")
        parser.add_argument("--cleanup", "-c", action="store_true", help="Whether to cleanup ")

        args = parser.parse_args()

        cleanup = args.cleanup

        test_apps = read_yaml(args.yaml)

        print(f"test_apps = {test_apps}")

        site_launcher = SiteLauncher(poc_directory=args.poc, app_path=args.app_path)

        site_launcher.start_server()
        site_launcher.start_clients(n=args.n_clients)

        admin_controller = AdminController(app_path=args.app_path)
        admin_controller.initialize()

        admin_controller.ensure_clients_started(num_clients=args.n_clients)

        print(f"Server status: {admin_controller.server_status()}.")

        app_results = []
        for app_data in test_apps:
            start_time = time.time()

            test_app, validators = app_data

            print(f"Running app {test_app} with {validators}")

            admin_controller.deploy_app(app_name=test_app)

            # time.sleep(2)

            print(f"Server status after app deployment: {admin_controller.server_status()}.")
            print(f"Client status after app deployment: {admin_controller.client_status()}")

            admin_controller.run_app()

            server_data = site_launcher.get_server_data()
            client_data = site_launcher.get_client_data()
            run_data = admin_controller.get_run_data()

            # Get the app validator
            if validators:
                validate_result = True
                for validator_module in validators:

                    # Create validator instance
                    module_name, class_name = get_module_class_from_full_path(validator_module)
                    app_validator_cls = getattr(importlib.import_module(module_name), class_name)
                    app_validator = app_validator_cls()

                    app_validate_res = app_validator.validate_results(
                        server_data=server_data,
                        client_data=client_data,
                        run_data=run_data,
                    )
                    validate_result = validate_result and app_validate_res

                app_results.append((test_app, validate_result))
            else:
                print(f"No validators provided so results can't be checked.")

            print(f"Finished running {test_app} in {time.time()-start_time} seconds.")

        print(f"App results: {app_results}")
        failure = False
        for app_name, app_result in app_results:
            print(f"App name: {app_name}, Result: {app_result}")
            if not app_result:
                failure = True

        if failure:
            sys.exit(1)
    except BaseException as e:
        traceback.print_exc()
        print(f"Exception in test run: {e.__str__()}")
        raise ValueError("Tests failed") from e
    finally:
        if site_launcher:
            site_launcher.stop_all_sites()

            if cleanup:
                site_launcher.finalize()
