# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from src.simple_controller import SimpleController
from src.simple_executor import SimpleExecutor
from src.standalone_file_streaming import FileReceiver, FileSender

from nvflare import FedJob


def main():
    # Create the FedJob
    job = FedJob(name="simple_file_streaming", min_clients=1)

    # Define the controller workflow, and file receiver
    # and send to server
    controller = SimpleController()
    receiver = FileReceiver()
    job.to_server(controller)
    job.to_server(receiver, id="receiver")

    # Define the executor, and file sender
    # and send to clients
    executor = SimpleExecutor()
    sender = FileSender()
    job.to_clients(executor, tasks=["train"])
    job.to_clients(sender, id="sender")

    # Export the job
    job_dir = "/tmp/nvflare/workspace/jobs/simple_file_streaming"
    print("job_dir=", job_dir)
    job.export_job(job_dir)

    # Run the job
    work_dir = "/tmp/nvflare/workspace/works/simple_file_streaming"
    print("workspace_dir=", work_dir)

    # starting the monitoring
    job.simulator_run(work_dir, n_clients=1, threads=1)


if __name__ == "__main__":
    main()
