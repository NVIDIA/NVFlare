from nvflare.fuel.hci.client.cli import AdminClient, CredentialType
from nvflare.fuel.hci.client.rr_service_finder import RRServiceFinder
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
    parser.add_argument('--port2', type=int, default=55551, required=False)
    args = parser.parse_args()

    ft_module = FileTransferModule(
        upload_dir='/Users/yanc/dlmed/client_up',
        download_dir='/Users/yanc/dlmed/client_down',
    )

    print('Admin Server: {} on port {}'.format(args.host, args.port))
    client = AdminClient(
        prompt='FLARE > ',
        credential_type=CredentialType.CERT,
        cmd_modules=[ft_module, SessionModule()],
        service_finder=RRServiceFinder(
            change_interval=20,
            host1=args.host,
            port1=args.port,
            host2=args.host,
            port2=args.port2),
        ca_cert="/Users/yanc/certs/rootCA.pem",
        client_cert="/Users/yanc/certs/client.crt",
        client_key="/Users/yanc/certs/client.key",
        debug=True
    )

    client.run()


if __name__ == "__main__":
    main()
