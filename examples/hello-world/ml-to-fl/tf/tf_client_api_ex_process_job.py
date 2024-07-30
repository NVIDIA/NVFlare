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


from src.tf_net import TFNet

from nvflare import FedAvg, FedJob
from nvflare.app_common.executors.client_api_launcher_executor import ClientAPILauncherExecutor
from nvflare.app_common.launchers.subprocess_launcher import SubprocessLauncher
from nvflare.app_opt.tf.params_converter import KerasModelToNumpyParamsConverter, NumpyToKerasModelParamsConverter
from nvflare.fuel.utils.pipe.file_pipe import FilePipe

if __name__ == "__main__":
    n_clients = 2
    num_rounds = 2
    train_script = "src/cifar10_tf_multi_gpu_fl.py"

    job = FedJob(name="tf_client_api_ex_process", min_clients=2)

    # Define the controller workflow and send to server
    controller = FedAvg(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )
    job.to(controller, "server")

    # Define the initial global model and send to server
    job.to(TFNet(input_shape=(None, 32, 32, 3)), "server")

    # Add clients
    for i in range(n_clients):
        executor = ClientAPILauncherExecutor(
            launcher_id=job.as_id(SubprocessLauncher(script=f"python3 custom/{train_script}", launch_once=True)),
            pipe_id=job.as_id(
                FilePipe(mode="PASSIVE", root_path=f"/tmp/nvflare/jobs/workdir/site-{i+1}/simulate_job/app_site-{i+1}")
            ),
            from_nvflare_converter_id=job.as_id(NumpyToKerasModelParamsConverter()),
            to_nvflare_converter_id=job.as_id(KerasModelToNumpyParamsConverter()),
            params_transfer_type="DIFF",
        )
        job.to(executor, f"site-{i+1}", gpu=0)
        job.to(train_script, f"site-{i+1}")
        job.to("src/tf_net.py", f"site-{i+1}")

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir")
