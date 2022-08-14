from nvflare.fuel.hci.client.cli import AdminClient, CredentialType
from nvflare.fuel.hci.client.static_service_finder import StaticServiceFinder
from nvflare.fuel.hci.client.file_transfer import FileTransferModule
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.client.api_spec import CommandContext

import argparse


class SessionModule(CommandModule):

    def get_spec(self):
        return CommandModuleSpec(
            name="sess",
            cmd_specs=[
                CommandSpec(
                    name="list_sessions",
                    description="list user sessions on server",
                    usage="list_sessions",
                    handler_func=self.list_sessions,
                    visible=True,
                )
            ])

    def list_sessions(self, args, ctx: CommandContext):
        api = ctx.get_api()
        return api.server_execute("list_sessions")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost', required=False)
    parser.add_argument('--port', type=int, default=55550, required=False)
    args = parser.parse_args()

    ft_module = FileTransferModule(
        upload_dir='/Users/yanc/dlmed/client_up',
        download_dir='/Users/yanc/dlmed/client_down',
    )

    print('Admin Server: {} on port {}'.format(args.host, args.port))
    client = AdminClient(
        prompt='CellNet > ',
        credential_type=CredentialType.PASSWORD,
        cmd_modules=[ft_module, SessionModule()],
        service_finder=StaticServiceFinder(args.host, args.port),
        ca_cert="/Users/yanc/certs/rootCA.pem",
        client_cert="/Users/yanc/certs/client.crt",
        client_key="/Users/yanc/certs/client.key",
        debug=False
    )

    client.run()


if __name__ == "__main__":
    main()
