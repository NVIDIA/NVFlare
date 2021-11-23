# Copyright (c) 2021, NVIDIA CORPORATION.
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

import os
import argparse
import time

from nvflare.fuel.hci.client.fl_admin_api import FLAdminAPI
from nvflare.fuel.hci.client.fl_admin_api_constants import FLDetailKey
from nvflare.fuel.hci.client.fl_admin_api_spec import TargetType


def api_command_wrapper(api_command_result):
    # prints the result of the command and raises RuntimeError to interrupt command sequence if there is an error
    print(api_command_result)
    if not api_command_result["status"] == "SUCCESS":
        raise RuntimeError("command was not successful!")

    return api_command_result


def wait_until_clients_gt2_cb(reply):
    # use as the callback in wait_until_server_status for waiting until a minimum of 2 clients registered on server
    if reply["details"][FLDetailKey.REGISTERED_CLIENTS] >= 2:
        return True
    else:
        return False


def attempt_login_until_success(api):
    logged_in = False
    while not logged_in:
        reply = None
        try:
            print("api.login()")
            reply = api_command_wrapper(api.login(username="admin@nvidia.com"))
        except RuntimeError:
            time.sleep(10)
            pass
        if reply and reply["status"] == "SUCCESS":
            logged_in = True
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_number", type=int, default=100, help="FL run number to start at.")
    parser.add_argument(
        "--admin_dir", type=str, default="/workspace/hello_nvflare/packages/admin", help="Path to admin directory."
    )

    args = parser.parse_args()

    host = ""
    port = 8003

    # Set up certificate names and admin folders
    ca_cert = os.path.join(args.admin_dir, "startup", "rootCA.pem")
    client_cert = os.path.join(args.admin_dir, "startup", "client.crt")
    client_key = os.path.join(args.admin_dir, "startup", "client.key")
    upload_dir = os.path.join(args.admin_dir, "transfer")
    download_dir = os.path.join(args.admin_dir, "download")
    if not os.path.isdir(download_dir):
        os.makedirs(download_dir)

    assert os.path.isdir(args.admin_dir), f"admin directory does not exist at {args.admin_dir}"
    assert os.path.isfile(ca_cert), f"rootCA.pem does not exist at {ca_cert}"
    assert os.path.isfile(client_cert), f"client.crt does not exist at {client_cert}"
    assert os.path.isfile(client_key), f"client.key does not exist at {client_key}"
    run_number = args.run_number

    # Connect with admin client
    api = FLAdminAPI(
        host=host,
        port=port,
        ca_cert=ca_cert,
        client_cert=client_cert,
        client_key=client_key,
        upload_dir=upload_dir,
        download_dir=download_dir,
        debug=False
    )
    reply = api.login(username="admin@nvidia.com")
    for k in reply.keys():
        assert "error" not in reply[k].lower(), f"Login not successful with {reply}"

    print("api.set_timeout(30)")
    api_command_wrapper(api.set_timeout(30))
    apps_to_deploy_and_run = ["hello_numpy_2", "hello_numpy_cross_val"]
    # With a list of apps_to_deploy_and_run to run consecutively, the assumption is that all apps are configured
    # properly to run without errors and all apps already exist in upload_dir. Custom code can be written to reconfigure
    # apps or handle errors that may occur.

    try:
        for app in apps_to_deploy_and_run:
            print(f'api.restart("{TargetType.ALL}")')
            api_command_wrapper(api.restart(TargetType.ALL))
            attempt_login_until_success(api)
            print("api.wait_until_server_status(callback=wait_until_clients_gt2_cb)")
            api_command_wrapper(api.wait_until_server_status(callback=wait_until_clients_gt2_cb))
            print("api.check_status(TargetType.SERVER)")
            api_command_wrapper(api.check_status(TargetType.SERVER))
            print(f"api.set_run_number({run_number})")
            api_command_wrapper(api.set_run_number(run_number))
            print(f'api.upload_app("{app}")')
            api_command_wrapper(api.upload_app(app))
            print(f'api.deploy_app("{app}", TargetType.ALL)')
            api_command_wrapper(api.deploy_app(app, TargetType.ALL))
            print("api.check_status(TargetType.CLIENT)")
            api_command_wrapper(api.check_status(TargetType.CLIENT))
            print("api.start_app(TargetType.ALL)")
            api_command_wrapper(api.start_app(TargetType.ALL))
            print("api.wait_until_server_stats()")
            wait_result = api_command_wrapper(api.wait_until_client_status(timeout=2000))
            if wait_result.get("details"):
                if wait_result.get("details").get("message"):
                    if wait_result.get("details").get("message") == "Waited until timeout.":
                        print("aborting because waited until timeout with clients still not stopped.")
                        print("api.abort(TargetType.ALL)")
                        api_command_wrapper(api.abort(TargetType.ALL))
            # api_command_wrapper("api.wait_until()", api.wait_until())
            # now server engine status should be stopped
            print("api.check_status(TargetType.SERVER)")
            api_command_wrapper(api.check_status(TargetType.SERVER))
            print("api.check_status(TargetType.CLIENT)")
            api_command_wrapper(api.check_status(TargetType.CLIENT))
            time.sleep(10)  # wait for clients to stop in case they take longer than server to stop
            print("api.check_status(TargetType.CLIENT)")
            api_command_wrapper(api.check_status(TargetType.CLIENT))
            run_number = run_number + 1

    except RuntimeError as e:
        print(f"There was an exception {e}.")

    # log out
    print("Admin logging out.")
    api.logout()


if __name__ == "__main__":
    main()
