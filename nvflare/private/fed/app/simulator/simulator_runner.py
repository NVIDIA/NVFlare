# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import json
import logging.config
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Process
from multiprocessing.connection import Client

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import JobConstants, MachineStatus, RunProcessKey, WorkspaceConstants
from nvflare.apis.job_def import ALL_SITES, JobMetaKey
from nvflare.apis.utils.job_utils import convert_legacy_zipped_app_to_job
from nvflare.apis.workspace import Workspace
from nvflare.fuel.common.multi_process_executor_constants import CommunicationMetaData
from nvflare.fuel.f3.mpm import MainProcessMonitor as mpm
from nvflare.fuel.f3.stats_pool import StatsPoolManager
from nvflare.fuel.hci.server.authz import AuthorizationService
from nvflare.fuel.sec.audit import AuditService
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.fuel.utils.network_utils import get_open_ports
from nvflare.fuel.utils.zip_utils import split_path, unzip_all_from_bytes, zip_directory_to_bytes
from nvflare.lighter.poc_commands import get_host_gpu_ids
from nvflare.private.defs import AppFolderConstants
from nvflare.private.fed.app.deployer.simulator_deployer import SimulatorDeployer
from nvflare.private.fed.app.utils import kill_child_processes
from nvflare.private.fed.client.client_status import ClientStatus
from nvflare.private.fed.server.job_meta_validator import JobMetaValidator
from nvflare.private.fed.simulator.simulator_app_runner import SimulatorServerAppRunner
from nvflare.private.fed.simulator.simulator_audit import SimulatorAuditor
from nvflare.private.fed.simulator.simulator_const import SimulatorConstants
from nvflare.private.fed.utils.fed_utils import add_logfile_handler, fobs_initialize
from nvflare.security.logging import secure_format_exception
from nvflare.security.security import EmptyAuthorizer

CLIENT_CREATE_POOL_SIZE = 200
POOL_STATS_DIR = "pool_stats"
SIMULATOR_POOL_STATS = "simulator_cell_stats.json"


class SimulatorRunner(FLComponent):
    def __init__(
        self, job_folder: str, workspace: str, clients=None, n_clients=None, threads=None, gpu=None, max_clients=100
    ):
        super().__init__()

        self.job_folder = job_folder
        self.workspace = workspace
        self.clients = clients
        self.n_clients = n_clients
        self.threads = threads
        self.gpu = gpu
        self.max_clients = max_clients

        self.ask_to_stop = False

        self.simulator_root = None
        self.server = None
        self.deployer = SimulatorDeployer()
        self.client_names = []
        self.federated_clients = []
        self.client_config = None
        self.deploy_args = None
        self.build_ctx = None

        self.clients_created = 0

    def _generate_args(
        self, job_folder: str, workspace: str, clients=None, n_clients=None, threads=None, gpu=None, max_clients=100
    ):
        args = Namespace(
            job_folder=job_folder,
            workspace=workspace,
            clients=clients,
            n_clients=n_clients,
            threads=threads,
            gpu=gpu,
            max_clients=max_clients,
        )
        args.set = []
        return args

    def setup(self):
        running_dir = os.getcwd()
        if self.workspace is None:
            self.workspace = "simulator_workspace"
            self.logger.warn(
                f"Simulator workspace is not provided. Set it to the default location:"
                f" {os.path.join(running_dir, self.workspace)}"
            )
        self.workspace = os.path.join(running_dir, self.workspace)

        self.args = self._generate_args(
            self.job_folder, self.workspace, self.clients, self.n_clients, self.threads, self.gpu, self.max_clients
        )

        if self.args.clients:
            self.client_names = self.args.clients.strip().split(",")
        else:
            if self.args.n_clients:
                for i in range(self.args.n_clients):
                    self.client_names.append("site-" + str(i + 1))

        log_config_file_path = os.path.join(self.args.workspace, "startup", WorkspaceConstants.LOGGING_CONFIG)
        if not os.path.isfile(log_config_file_path):
            log_config_file_path = os.path.join(os.path.dirname(__file__), WorkspaceConstants.LOGGING_CONFIG)
        logging.config.fileConfig(fname=log_config_file_path, disable_existing_loggers=False)
        local_dir = os.path.join(self.args.workspace, "local")
        os.makedirs(local_dir, exist_ok=True)
        shutil.copyfile(log_config_file_path, os.path.join(local_dir, WorkspaceConstants.LOGGING_CONFIG))

        self.args.log_config = None
        self.args.config_folder = "config"
        self.args.job_id = SimulatorConstants.JOB_NAME
        self.args.client_config = os.path.join(self.args.config_folder, JobConstants.CLIENT_JOB_CONFIG)
        self.args.env = os.path.join("config", AppFolderConstants.CONFIG_ENV)
        cwd = os.getcwd()
        self.args.job_folder = os.path.join(cwd, self.args.job_folder)

        if not os.path.exists(self.args.workspace):
            os.makedirs(self.args.workspace)
        os.chdir(self.args.workspace)
        fobs_initialize()
        AuthorizationService.initialize(EmptyAuthorizer())
        AuditService.the_auditor = SimulatorAuditor()

        self.simulator_root = os.path.join(self.args.workspace, SimulatorConstants.JOB_NAME)
        if os.path.exists(self.simulator_root):
            shutil.rmtree(self.simulator_root)

        os.makedirs(self.simulator_root)
        log_file = os.path.join(self.simulator_root, WorkspaceConstants.LOG_FILE_NAME)
        add_logfile_handler(log_file)

        try:
            data_bytes, job_name, meta = self.validate_job_data()

            if not self.client_names:
                self.client_names = self._extract_client_names_from_meta(meta)

            if not self.client_names:
                self.args.n_clients = 2
                self.logger.warn("The number of simulator clients is not provided. Setting it to default: 2")
                for i in range(self.args.n_clients):
                    self.client_names.append("site-" + str(i + 1))
            if self.args.gpu is None and self.args.threads is None:
                self.args.threads = 1
                self.logger.warn("The number of threads is not provided. Set it to default: 1")

            if self.max_clients < len(self.client_names):
                self.logger.error(
                    f"The number of clients ({len(self.client_names)}) can not be more than the "
                    f"max_number of clients ({self.max_clients})"
                )
                return False

            if self.args.gpu:
                gpus = self.args.gpu.split(",")
                host_gpus = [str(x) for x in (get_host_gpu_ids())]
                if host_gpus and not set(gpus).issubset(host_gpus):
                    wrong_gpus = [x for x in gpus if x not in host_gpus]
                    self.logger.error(f"These GPUs are not available: {wrong_gpus}")
                    return False

                if len(gpus) > len(self.client_names):
                    self.logger.error(
                        f"The number of clients ({len(self.client_names)}) must be larger than or equal to "
                        f"the number of GPUS: ({len(gpus)})"
                    )
                    return False
                if len(gpus) > 1:
                    if self.args.threads and self.args.threads > 1:
                        self.logger.info(
                            "When running with multi GPU, each GPU will run with only 1 thread. "
                            "Set the Threads to 1."
                        )
                    self.args.threads = 1
                elif len(gpus) == 1:
                    if self.args.threads is None:
                        self.args.threads = 1
                        self.logger.warn("The number of threads is not provided. Set it to default: 1")

            if self.args.threads and self.args.threads > len(self.client_names):
                self.logger.error("The number of threads to run can not be larger than the number of clients.")
                return False
            if not (self.args.gpu or self.args.threads):
                self.logger.error("Please provide the number of threads or provide gpu options to run the simulator.")
                return False

            self._validate_client_names(meta, self.client_names)

            # Deploy the FL server
            self.logger.info("Create the Simulator Server.")
            simulator_server, self.server = self.deployer.create_fl_server(self.args)
            # self.services.deploy(self.args, grpc_args=simulator_server)

            self.logger.info("Deploy the Apps.")
            self._deploy_apps(job_name, data_bytes, meta)

            return True

        except Exception as e:
            self.logger.error(f"Simulator setup error: {secure_format_exception(e)}")
            return False

    def validate_job_data(self):
        # Validate the simulate job
        job_name = split_path(self.args.job_folder)[1]
        data = zip_directory_to_bytes("", self.args.job_folder)
        data_bytes = convert_legacy_zipped_app_to_job(data)
        job_validator = JobMetaValidator()
        valid, error, meta = job_validator.validate(job_name, data_bytes)
        if not valid:
            raise RuntimeError(error)
        return data_bytes, job_name, meta

    def _extract_client_names_from_meta(self, meta):
        client_names = []
        for _, participants in meta.get(JobMetaKey.DEPLOY_MAP).items():
            for p in participants:
                if p.upper() != ALL_SITES and p != "server":
                    client_names.append(p)
        return client_names

    def _validate_client_names(self, meta, client_names):
        no_app_clients = []
        for name in client_names:
            name_matched = False
            for app_name, participants in meta.get(JobMetaKey.DEPLOY_MAP).items():
                if len(participants) == 1 and participants[0].upper() == ALL_SITES:
                    name_matched = True
                    break
                if name in participants:
                    name_matched = True
                    break
            if not name_matched:
                no_app_clients.append(name)
        if no_app_clients:
            raise RuntimeError(f"The job does not have App to run for clients: {no_app_clients}")

    def _deploy_apps(self, job_name, data_bytes, meta):
        with tempfile.TemporaryDirectory() as temp_dir:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.mkdir(temp_dir)
            unzip_all_from_bytes(data_bytes, temp_dir)
            temp_job_folder = os.path.join(temp_dir, job_name)

            app_server_root = os.path.join(self.simulator_root, "app_server")
            for app_name, participants in meta.get(JobMetaKey.DEPLOY_MAP).items():
                if len(participants) == 1 and participants[0].upper() == ALL_SITES:
                    participants = ["server"]
                    participants.extend([client for client in self.client_names])

                for p in participants:
                    if p == "server":
                        app = os.path.join(temp_job_folder, app_name)
                        shutil.copytree(app, app_server_root)
                    elif p in self.client_names:
                        app_client_root = os.path.join(self.simulator_root, "app_" + p)
                        app = os.path.join(temp_job_folder, app_name)
                        shutil.copytree(app, app_client_root)

    def split_clients(self, clients: [], gpus: []):
        split_clients = []
        for _ in gpus:
            split_clients.append([])
        index = 0
        for client in clients:
            split_clients[index % len(gpus)].append(client)
            index += 1
        return split_clients

    def create_clients(self):
        # Deploy the FL clients
        self.logger.info("Create the simulate clients.")
        clients_created_waiter = threading.Event()
        for client_name in self.client_names:
            self.create_client(client_name)

        self.logger.info("Set the client status ready.")
        self._set_client_status()

    def create_client(self, client_name):
        client, self.client_config, self.deploy_args, self.build_ctx = self.deployer.create_fl_client(
            client_name, self.args
        )
        self.federated_clients.append(client)
        app_root = os.path.join(self.simulator_root, "app_" + client_name)
        app_custom_folder = os.path.join(app_root, "custom")
        sys.path.append(app_custom_folder)

    def _set_client_status(self):
        for client in self.federated_clients:
            app_client_root = os.path.join(self.simulator_root, "app_" + client.client_name)
            client.app_client_root = app_client_root
            client.args = self.args
            # self.create_client_runner(client)
            client.simulate_running = False
            client.status = ClientStatus.STARTED

    def run(self):
        try:
            manager = Manager()
            return_dict = manager.dict()
            process = Process(target=self.run_processs, args=(return_dict,))
            process.start()
            process.join()
            if process.exitcode == -9:
                run_status = process.exitcode
            else:
                run_status = return_dict["run_status"]
            return run_status
        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt, terminate all the child processes.")
            kill_child_processes(os.getpid())
            return -9

    def run_processs(self, return_dict):
        # run_status = self.simulator_run_main()
        run_status = mpm.run(main_func=self.simulator_run_main, shutdown_grace_time=3, cleanup_grace_time=6)

        return_dict["run_status"] = run_status

    def simulator_run_main(self):
        if self.setup():
            try:
                self.create_clients()
                self.server.engine.run_processes[SimulatorConstants.JOB_NAME] = {
                    RunProcessKey.LISTEN_PORT: None,
                    RunProcessKey.CONNECTION: None,
                    RunProcessKey.CHILD_PROCESS: None,
                    RunProcessKey.JOB_ID: SimulatorConstants.JOB_NAME,
                    RunProcessKey.PARTICIPANTS: self.server.engine.client_manager.clients,
                }

                self.logger.info("Deploy and start the Server App.")
                server_thread = threading.Thread(target=self.start_server_app, args=[])
                server_thread.start()

                # wait for the server app is started
                while self.server.engine.engine_info.status != MachineStatus.STARTED:
                    time.sleep(1.0)
                    if not server_thread.is_alive():
                        raise RuntimeError("Could not start the Server App.")

                # # Start the client heartbeat calls.
                # for client in self.federated_clients:
                #     client.start_heartbeat(interval=2)

                if self.args.gpu:
                    gpus = self.args.gpu.split(",")
                    split_clients = self.split_clients(self.federated_clients, gpus)
                else:
                    gpus = [None]
                    split_clients = [self.federated_clients]

                executor = ThreadPoolExecutor(max_workers=len(gpus))
                for index in range(len(gpus)):
                    clients = split_clients[index]
                    executor.submit(lambda p: self.client_run(*p), [clients, gpus[index]])

                executor.shutdown()
                # Abort the server after all clients finished run
                self.server.abort_run()
                server_thread.join()
                run_status = 0
            except Exception as e:
                self.logger.error(f"Simulator run error: {secure_format_exception(e)}")
                run_status = 2
            finally:
                # self.services.close()
                self.deployer.close()
        else:
            run_status = 1
        return run_status

    def client_run(self, clients, gpu):
        client_runner = SimulatorClientRunner(self.args, clients, self.client_config, self.deploy_args, self.build_ctx)
        client_runner.run(gpu)

    def start_server_app(self):
        app_server_root = os.path.join(self.simulator_root, "app_server")
        self.args.server_config = os.path.join("config", JobConstants.SERVER_JOB_CONFIG)
        app_custom_folder = os.path.join(app_server_root, "custom")
        sys.path.append(app_custom_folder)

        startup = os.path.join(self.args.workspace, WorkspaceConstants.STARTUP_FOLDER_NAME)
        os.makedirs(startup, exist_ok=True)
        local = os.path.join(self.args.workspace, WorkspaceConstants.SITE_FOLDER_NAME)
        os.makedirs(local, exist_ok=True)
        workspace = Workspace(root_dir=self.args.workspace, site_name="server")

        self.server.job_cell = self.server.create_job_cell(
            SimulatorConstants.JOB_NAME,
            self.server.cell.get_root_url_for_child(),
            self.server.cell.get_internal_listener_url(),
            False,
            None,
        )
        server_app_runner = SimulatorServerAppRunner(self.server)
        snapshot = None
        server_app_runner.start_server_app(
            workspace, self.args, app_server_root, self.args.job_id, snapshot, self.logger
        )

        # start = time.time()
        # while self.services.engine.client_manager.clients:
        #     # Wait for the clients to shutdown and quite first.
        #     time.sleep(0.1)
        #     if time.time() - start > 30.:
        #         break

        self.dump_stats(workspace)

        self.server.admin_server.stop()
        self.server.close()

    def dump_stats(self, workspace: Workspace):
        stats_dict = StatsPoolManager.to_dict()
        json_object = json.dumps(stats_dict, indent=4)
        os.makedirs(os.path.join(workspace.get_run_dir(SimulatorConstants.JOB_NAME), POOL_STATS_DIR))
        file = os.path.join(workspace.get_run_dir(SimulatorConstants.JOB_NAME), POOL_STATS_DIR, SIMULATOR_POOL_STATS)
        with open(file, "w") as outfile:
            outfile.write(json_object)


class SimulatorClientRunner(FLComponent):
    def __init__(self, args, clients: [], client_config, deploy_args, build_ctx):
        super().__init__()
        self.args = args
        self.federated_clients = clients
        self.run_client_index = -1

        self.simulator_root = os.path.join(self.args.workspace, SimulatorConstants.JOB_NAME)
        self.client_config = client_config
        self.deploy_args = deploy_args
        self.build_ctx = build_ctx
        self.kv_list = parse_vars(args.set)

    def run(self, gpu):
        try:
            # self.create_clients()
            self.logger.info("Start the clients run simulation.")
            executor = ThreadPoolExecutor(max_workers=self.args.threads)
            lock = threading.Lock()
            timeout = self.kv_list.get("simulator_worker_timeout", 60.0)
            for i in range(self.args.threads):
                executor.submit(lambda p: self.run_client_thread(*p), [self.args.threads, gpu, lock, timeout])

            # wait for the server and client running thread to finish.
            executor.shutdown()
        except Exception as e:
            self.logger.error(f"SimulatorClientRunner run error: {secure_format_exception(e)}")
        finally:
            for client in self.federated_clients:
                threading.Thread(target=self._shutdown_client, args=[client]).start()

    def _shutdown_client(self, client):
        try:
            client.communicator.heartbeat_done = True
            # time.sleep(3)
            client.terminate()
            # client.close()
            client.status = ClientStatus.STOPPED
            client.communicator.cell.stop()
        except:
            # Ignore the exception for the simulator client shutdown
            self.logger.warn(f"Exception happened to client{client.name} during shutdown ")

    def run_client_thread(self, num_of_threads, gpu, lock, timeout=60):
        stop_run = False
        interval = 1
        client_to_run = None  # indicates the next client to run

        try:
            while not stop_run:
                time.sleep(interval)
                with lock:
                    if not client_to_run:
                        client = self.get_next_run_client()
                    else:
                        client = client_to_run

                client.simulate_running = True
                stop_run, client_to_run = self.do_one_task(client, num_of_threads, gpu, lock, timeout=timeout)

                client.simulate_running = False
        except Exception as e:
            self.logger.error(f"run_client_thread error: {secure_format_exception(e)}")

    def do_one_task(self, client, num_of_threads, gpu, lock, timeout=60.0):
        open_port = get_open_ports(1)[0]
        command = (
            sys.executable
            + " -m nvflare.private.fed.app.simulator.simulator_worker -o "
            + self.args.workspace
            + " --client "
            + client.client_name
            + " --token "
            + client.token
            + " --port "
            + str(open_port)
            + " --parent_pid "
            + str(os.getpid())
            + " --simulator_root "
            + self.simulator_root
            + " --root_url "
            + str(client.cell.get_root_url_for_child())
            + " --parent_url "
            + str(client.cell.get_internal_listener_url())
        )
        if gpu:
            command += " --gpu " + str(gpu)
        _ = subprocess.Popen(shlex.split(command, True), preexec_fn=os.setsid, env=os.environ.copy())

        conn = self._create_connection(open_port, timeout=timeout)

        self.build_ctx["client_name"] = client.client_name
        data = {
            # SimulatorConstants.CLIENT: client,
            SimulatorConstants.CLIENT_CONFIG: self.client_config,
            SimulatorConstants.DEPLOY_ARGS: self.deploy_args,
            SimulatorConstants.BUILD_CTX: self.build_ctx,
        }
        conn.send(data)

        while True:
            stop_run = conn.recv()

            with lock:
                if num_of_threads != len(self.federated_clients):
                    next_client = self.get_next_run_client()
                else:
                    next_client = client
            if not stop_run and next_client.client_name == client.client_name:
                conn.send(True)
            else:
                conn.send(False)
                break

        return stop_run, next_client

    def _create_connection(self, open_port, timeout=60.0):
        conn = None
        start = time.time()
        while not conn:
            try:
                address = ("localhost", open_port)
                conn = Client(address, authkey=CommunicationMetaData.CHILD_PASSWORD.encode())
            except Exception:
                if time.time() - start > timeout:
                    raise RuntimeError(
                        f"Failed to create connection to the child process in {self.__class__.__name__},"
                        f" timeout: {timeout}"
                    )
                time.sleep(1.0)
                pass
        return conn

    def get_next_run_client(self):
        # Find the next client which is not currently running
        while True:
            self.run_client_index = (self.run_client_index + 1) % len(self.federated_clients)
            client = self.federated_clients[self.run_client_index]
            if not client.simulate_running:
                break
        self.logger.info(f"Simulate Run client: {client.client_name}")
        return client
