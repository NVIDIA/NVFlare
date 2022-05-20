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
from nvflare.fuel.hci.client.fl_admin_api import FLAdminAPI


def run_admin_api_tests(admin_api: FLAdminAPI):
    print(("\n" + "*" * 120) * 20)
    print("\n" + "=" * 40)
    print("\nRunning through tests of admin commands:")
    print("\n" + "=" * 40)
    print("\nCommand: set_timeout")
    print(admin_api.set_timeout(11).get("details").get("message"))
    print("\nActive SP:")
    print(admin_api.get_active_sp().get("details"))
    print("\nList SP:")
    print(admin_api.list_sp().get("details"))
    print("\nCommand: get_available_apps_to_upload")
    print(admin_api.get_available_apps_to_upload())
    print("\nList Jobs:")
    list_jobs_return_message = admin_api.list_jobs().get("details").get("message")
    print(list_jobs_return_message)
    first_job = list_jobs_return_message.split()[14]
    print("\nCommand: ls server -a .")
    ls_return_message = admin_api.ls_target("server", "-a", ".").get("details").get("message")
    print(ls_return_message)
    print("\nAssert Job {} is in the server root dir...".format(first_job))
    assert(first_job in ls_return_message)

    print("\nAborting Job {}:".format(first_job))
    print("\n" + "=" * 50)
    print(admin_api.abort_job(first_job).get("details").get("message"))
    print("\n" + "=" * 50)
    # print("\nCloning Job {}:".format(first_job))
    # clone_job_return = admin_api.clone_job(first_job)
    # print(clone_job_return.get("details").get("message"))
    # second_job = clone_job_return.get("details").get("job_id")
    # print("\nSecond job_id is: {}".format(second_job))
    print("\nCommand: env server")
    print(admin_api.env_target("server").get("details"))
    print("\nCommand: pwd")
    print(admin_api.get_working_directory("server").get("details").get("message"))
    print("\nCommand: env site-1")
    print(admin_api.env_target("site-1").get("details"))
    print("\nCommand: tail_target_log server")
    tail_return_message = admin_api.tail_target_log("server").get("details").get("message")
    print(tail_return_message)
    print("\nAssert first job matches end of tail...".format(first_job))
    assert(tail_return_message[-36:] == first_job)
    print("\nCommand: grep_target server -n 'deployed to the server for run' log.txt")
    grep_return_message = (
        admin_api.grep_target("server", "-n", "Stop the job run", "log.txt")
        .get("details")
        .get("message")
    )
    print(grep_return_message)
    print("\nAssert first job matches job_id in grep...".format(first_job))
    assert(grep_return_message[-36:] == first_job)
    print("\nDeleting run for the first job {}:".format(first_job))
    print(admin_api.delete_run(first_job))
    print("\nCommand: ls server -a .")
    ls_return_message = admin_api.ls_target("server", "-a", ".").get("details").get("message")
    print(ls_return_message)
    print("\nAssert Job {} is no longer in the server root dir...".format(first_job))
    assert(first_job not in ls_return_message)

    # print("\nAborting Job {}:".format(second_job))
    print("\n" + "=" * 50)
    # print(admin_api.abort_job(second_job).get("details").get("message"))
    print("Finished with admin commands testing through FLAdminAPI.")
    print(("\n" + "*" * 120) * 20)