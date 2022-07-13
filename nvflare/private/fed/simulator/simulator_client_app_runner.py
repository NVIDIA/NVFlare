from nvflare.private.fed.client.client_app_runner import ClientAppRunner
from nvflare.private.fed.server.server_app_runner import ServerAppRunner


class SimulatorClientAppRunner(ClientAppRunner):

    def start_command_agent(self, args, client_runner, federated_client, fl_ctx):
        pass


class SimulatorServerAppRunner(ServerAppRunner):

    def sync_up_parents_process(self, args, server):
        pass
