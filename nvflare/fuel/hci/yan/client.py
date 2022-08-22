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
    parser.add_argument('--ssl', action='store_true')
    args = parser.parse_args()

    ft_module = FileTransferModule(
        upload_dir='/Users/yanc/dlmed/client_up',
        download_dir='/Users/yanc/dlmed/client_down',
    )

    if args.ssl:
        cred_type = CredentialType.CERT
        print("Start client with SSL: user cert to login")
    else:
        cred_type = CredentialType.PASSWORD
        print("Start client without SSL: user pwd to login")

    client = AdminClient(
        prompt='FLARE > ',
        credential_type=cred_type,
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
        debug=False,
        session_timeout_interval=30
    )

    client.run()


if __name__ == "__main__":
    main()
