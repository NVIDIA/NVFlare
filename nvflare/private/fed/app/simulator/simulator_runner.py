import json
import os
import shutil
import sys
import threading

from nvflare.apis.fl_constant import MachineStatus
from nvflare.apis.job_def import JobMetaKey
from nvflare.private.defs import AppFolderConstants
from nvflare.private.fed.server.server_engine import ServerEngine
from nvflare.private.fed.server.server_json_config import ServerJsonConfigurator
from nvflare.private.fed.server.server_status import ServerStatus


class SimulatorRunner:

    def run(self, simulator_root, args, logger, services):
        meta_file = os.path.join(args.job_folder, "meta.json")
        with open(meta_file, "rb") as f:
            meta_data = f.read()
        meta = json.loads(meta_data)

        threading.Thread(target=self.start_server, args=[simulator_root, args, logger, services, meta]).start()

        threading.Thread(target=self.start_client, args=[simulator_root, args, logger, services, meta]).start()

    def start_server(self, simulator_root, args, logger, services, meta):
        # jid = str(uuid.uuid4())
        # meta[JobMetaKey.JOB_ID.value] = jid
        # meta[JobMetaKey.SUBMIT_TIME.value] = time.time()
        # meta[JobMetaKey.SUBMIT_TIME_ISO.value] = (
        #     datetime.datetime.fromtimestamp(meta[JobMetaKey.SUBMIT_TIME]).astimezone().isoformat()
        # )
        # meta[JobMetaKey.START_TIME.value] = ""
        # meta[JobMetaKey.DURATION.value] = "N/A"
        # meta[JobMetaKey.STATUS.value] = RunStatus.SUBMITTED.value
        app_server_root = os.path.join(simulator_root, "app_server")
        for app_name, participants in meta.get(JobMetaKey.DEPLOY_MAP).items():
            for p in participants:
                if p == "server":
                    app = os.path.join(args.job_folder, app_name)
                    shutil.copytree(app, app_server_root)
        snapshot = None
        self.start_server_app(services, args, app_server_root, "simulate_job", snapshot, logger)

    def start_server_app(self, server, args, app_root, job_id, snapshot, logger):

        try:
            server_config = os.path.join("config", AppFolderConstants.CONFIG_FED_SERVER)
            server_config_file_name = os.path.join(app_root, server_config)
            app_custom_folder = os.path.join(app_root, "custom")
            sys.path.append(app_custom_folder)

            conf = ServerJsonConfigurator(
                config_file_name=server_config_file_name,
            )
            conf.configure()

            self.set_up_run_config(server, conf)

            if not isinstance(server.engine, ServerEngine):
                raise TypeError(f"server.engine must be ServerEngine. Got type:{type(server.engine).__name__}")
            # server.engine.create_parent_connection(int(args.conn))
            # server.engine.sync_clients_from_main_process()

            server.start_run(job_id, app_root, conf, args, snapshot)
        except BaseException as e:
            logger.exception(f"FL server execution exception: {e}", exc_info=True)
            raise e
        finally:
            server.status = ServerStatus.STOPPED
            server.engine.engine_info.status = MachineStatus.STOPPED
            server.stop_training()

    def set_up_run_config(self, server, conf):
        server.heart_beat_timeout = conf.heartbeat_timeout
        server.runner_config = conf.runner_config
        server.handlers = conf.handlers

    def start_client(self, simulator_root, args, logger, services, meta):
        # jid = str(uuid.uuid4())
        # meta[JobMetaKey.JOB_ID.value] = jid
        # meta[JobMetaKey.SUBMIT_TIME.value] = time.time()
        # meta[JobMetaKey.SUBMIT_TIME_ISO.value] = (
        #     datetime.datetime.fromtimestamp(meta[JobMetaKey.SUBMIT_TIME]).astimezone().isoformat()
        # )
        # meta[JobMetaKey.START_TIME.value] = ""
        # meta[JobMetaKey.DURATION.value] = "N/A"
        # meta[JobMetaKey.STATUS.value] = RunStatus.SUBMITTED.value
        for app_name, participants in meta.get(JobMetaKey.DEPLOY_MAP).items():
            for p in participants:
                if p != "server":
                    app_client_root = os.path.join(simulator_root, "app_" + p)
                    app = os.path.join(args.job_folder, app_name)
                    shutil.copytree(app, app_client_root)
        snapshot = None
        self.start_server_app(services, args, app_client_root, "simulate_job", snapshot, logger)
