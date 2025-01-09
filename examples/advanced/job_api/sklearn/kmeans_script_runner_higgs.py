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

import csv
import os

from src.kmeans_assembler import KMeansAssembler
from src.split_csv import distribute_header_file, split_csv

from nvflare import FedJob
from nvflare.app_common.aggregators.collect_and_assemble_aggregator import CollectAndAssembleAggregator
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.sklearn.joblib_model_param_persistor import JoblibModelParamPersistor
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

preprocess = True  # if False, assume data is already preprocessed and split


def split_higgs(input_data_path, input_header_path, output_dir, site_num, sample_rate, site_name_prefix="site-"):
    input_file = input_data_path
    output_directory = output_dir
    num_parts = site_num
    site_name_prefix = site_name_prefix
    sample_rate = sample_rate
    split_csv(input_file, output_directory, num_parts, site_name_prefix, sample_rate)
    distribute_header_file(input_header_path, output_directory, num_parts, site_name_prefix)


if __name__ == "__main__":
    n_clients = 3
    num_rounds = 2
    train_script = "src/kmeans_fl.py"
    data_input_dir = "/tmp/nvflare/higgs/data"
    data_output_dir = "/tmp/nvflare/higgs/split_data"

    # Download data
    os.makedirs(data_input_dir, exist_ok=True)
    higgs_zip_file = os.path.join(data_input_dir, "higgs.zip")
    if not os.path.exists(higgs_zip_file):
        os.system(
            f"curl -o {higgs_zip_file} https://archive.ics.uci.edu/static/public/280/higgs.zip"
        )  # This might take a while. The file is 2.8 GB.
        os.system(f"unzip -d {data_input_dir} {higgs_zip_file}")
        os.system(
            f"gunzip -c {os.path.join(data_input_dir, 'HIGGS.csv.gz')} > {os.path.join(data_input_dir, 'higgs.csv')}"
        )

    if preprocess:  # if False, assume data is already preprocessed and split
        # Generate the csv header file
        # Your list of data
        features = [
            "label",
            "lepton_pt",
            "lepton_eta",
            "lepton_phi",
            "missing_energy_magnitude",
            "missing_energy_phi",
            "jet_1_pt",
            "jet_1_eta",
            "jet_1_phi",
            "jet_1_b_tag",
            "jet_2_pt",
            "jet_2_eta",
            "jet_2_phi",
            "jet_2_b_tag",
            "jet_3_pt",
            "jet_3_eta",
            "jet_3_phi",
            "jet_3_b_tag",
            "jet_4_pt",
            "jet_4_eta",
            "jet_4_phi",
            "jet_4_b_tag",
            "m_jj",
            "m_jjj",
            "m_lv",
            "m_jlv",
            "m_bb",
            "m_wbb",
            "m_wwbb",
        ]

        # Specify the file path
        file_path = os.path.join(data_input_dir, "headers.csv")

        with open(file_path, "w", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(features)

        print(f"features written to {file_path}")

        # Split the data
        split_higgs(
            input_data_path=os.path.join(data_input_dir, "higgs.csv"),
            input_header_path=os.path.join(data_input_dir, "headers.csv"),
            output_dir=data_output_dir,
            site_num=n_clients,
            sample_rate=0.3,
        )

    # Create the federated learning job
    job = FedJob(name="kmeans")

    # ScatterAndGather also expects an "aggregator" which we define here.
    # The actual aggregation function is defined by an "assembler" to specify how to handle the collected updates.
    # We use KMeansAssembler which is the assembler designed for k-Means algorithm.
    assembler_id = job.to_server(KMeansAssembler(), id="assembler")
    aggregator_id = job.to_server(CollectAndAssembleAggregator(assembler_id=assembler_id), id="aggregator")

    # For kmeans with sklean, we need a custom persistor
    # JoblibModelParamPersistor is a persistor which save/read the model to/from file with JobLib format.
    persistor_id = job.to_server(JoblibModelParamPersistor(initial_params={"n_clusters": 2}), id="persistor")

    shareable_generator_id = job.to_server(FullModelShareableGenerator(), id="shareable_generator")

    controller = ScatterAndGather(
        min_clients=n_clients,
        num_rounds=num_rounds,
        wait_time_after_min_received=0,
        aggregator_id=aggregator_id,
        persistor_id=persistor_id,
        shareable_generator_id=shareable_generator_id,
        train_task_name="train",  # Client will start training once received such task.
        train_timeout=0,
    )
    job.to(controller, "server")

    job.to(IntimeModelSelector(key_metric="accuracy"), "server")

    # Add clients
    for i in range(n_clients):
        executor = ScriptRunner(
            script=train_script,
            script_args=f"--data_root_dir {data_output_dir}",
            framework=FrameworkType.RAW,  # kmeans requires raw values only rather than PyTorch Tensors (the default)
        )
        job.to(executor, f"site-{i + 1}")  # HIGGs data splitter assumes site names start from 1

    # job.export_job("/tmp/nvflare/jobs/job_config")
    job.simulator_run("/tmp/nvflare/jobs/workdir")
