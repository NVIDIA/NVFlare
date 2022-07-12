from nvflare.apis.event_type import EventType
from .server_deployer import ServerDeployer
from .base_client_deployer import BaseClientDeployer
from nvflare.private.fed.simulator.simulator_server import SimulatorServer
from nvflare.private.fed.client.admin import FedAdminAgent
from nvflare.private.fed.client.admin_msg_sender import AdminMessageSender
from nvflare.private.fed.client.client_engine import ClientEngine
from nvflare.private.fed.client.fed_client import FederatedClient
# from nvflare.private.fed.simulator.simulator_job_runner import SimulatorJobRunner


class SimulatorDeploy(ServerDeployer):
    def create_fl_server(self, args, secure_train=False):
        simulator_server = self._create_simulator_server_config()

        wait_after_min_clients = simulator_server.get("wait_after_min_clients", 10)
        heart_beat_timeout = 600
        if simulator_server["heart_beat_timeout"]:
            heart_beat_timeout = simulator_server["heart_beat_timeout"]

        if self.host:
            target = simulator_server["service"].get("target", None)
            simulator_server["service"]["target"] = self.host + ":" + target.split(":")[1]

        services = SimulatorServer(
            project_name=simulator_server.get("name", ""),
            min_num_clients=simulator_server.get("min_num_clients", 1),
            max_num_clients=simulator_server.get("max_num_clients", 100),
            wait_after_min_clients=wait_after_min_clients,
            cmd_modules=self.cmd_modules,
            heart_beat_timeout=heart_beat_timeout,
            args=args,
            secure_train=secure_train,
            enable_byoc=self.enable_byoc,
            snapshot_persistor=self.snapshot_persistor,
            overseer_agent=self.overseer_agent,
        )
        return simulator_server, services

    def create_fl_client(self, args):
        client_config, build_ctx = self._create_simulator_client_config()

        deployer = BaseClientDeployer()
        deployer.build(build_ctx)
        federated_client = deployer.create_fed_client(args)

        federated_client.register()
        federated_client.start_heartbeat()
        servers = [{t["name"]: t["service"]} for t in client_config.get("servers")]
        admin_agent = self.create_admin_agent(
            sorted(servers)[0],
            federated_client,
            args,
        )
        admin_agent.start()

        return federated_client

    def create_admin_agent(self, server_args, federated_client: FederatedClient,args, rank=0):
        sender = AdminMessageSender(
            client_name=federated_client.token,
            server_args=server_args,
            secure=False,
        )
        client_engine = ClientEngine(federated_client, federated_client.token, sender, args, rank)
        admin_agent = FedAdminAgent(
            client_name="admin_agent",
            sender=sender,
            app_ctx=client_engine,
        )
        admin_agent.app_ctx.set_agent(admin_agent)
        federated_client.set_client_engine(client_engine)

        client_engine.fire_event(EventType.SYSTEM_START, client_engine.new_context())

        return admin_agent

    @staticmethod
    def _create_simulator_server_config():
        simulator_server = {
            "name": "simulator",
            "service": {
                "target": "localhost:6002",
                "options": [
                    [
                        "grpc.max_send_message_length",
                        2147483647
                    ],
                    [
                        "grpc.max_receive_message_length",
                        2147483647
                    ]
                ]
            },
            "admin_host": "localhost",
            "admin_port": 6003,
            "max_num_clients": 100,
            "heart_beat_timeout": 600,
            "num_server_workers": 4,
            "compression": "Gzip",
            "admin_storage": "transfer",
            "download_job_url": "http://download.server.com/",
            "min_num_clients": 1
        }
        return simulator_server

    @staticmethod
    def _create_simulator_client_config():
        client_config = {
            "servers": [
                {
                    "name": "simulator",
                    "service": {
                        "target": "localhost:6002",
                        "options": [
                            [
                                "grpc.max_send_message_length",
                                2147483647
                            ],
                            [
                                "grpc.max_receive_message_length",
                                2147483647
                            ]
                        ]
                    }
                }
            ],
            "client": {
                "retry_timeout": 30,
                "compression": "Gzip"
            }
        }

        build_ctx = {
            "client_name": "client1",
            "server_config": client_config.get("servers", []),
            "client_config": client_config["client"],
            "server_host": None,
            "secure_train": False,
            "enable_byoc": True,
            "overseer_agent": None,
            "client_components": {},
            "client_handlers": None,
        }

        return client_config, build_ctx

    # def create_job_runner(self, args):
    #     job_runner = SimulatorJobRunner(workspace_root=args.workspace)
    #     return job_runner

