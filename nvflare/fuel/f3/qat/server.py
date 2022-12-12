from nvflare.fuel.hci.server.hci import AdminServer
from nvflare.fuel.f3.cellnet import FQCN
from .cell_runner import CellRunner, NetConfig

from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.server.builtin import new_command_register_with_builtin_module
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.hci.server.login import LoginModule, SessionManager, SimpleAuthenticator
from nvflare.fuel.hci.security import hash_password


class Server(CellRunner, CommandModule):

    def __init__(
            self,
            net_config: NetConfig,
    ):
        admin_host, admin_port = net_config.get_admin()
        if not admin_host or not admin_port:
            raise RuntimeError("missing admin host/port in net config")

        users = {"admin": hash_password("admin")}
        cmd_reg = new_command_register_with_builtin_module(app_ctx=self)
        authenticator = SimpleAuthenticator(users)
        sess_mgr = SessionManager()
        login_module = LoginModule(authenticator, sess_mgr)
        cmd_reg.register_module(login_module)
        cmd_reg.register_module(sess_mgr)
        cmd_reg.register_module(self)

        self.admin = AdminServer(
            cmd_reg=cmd_reg,
            host=admin_host,
            port=admin_port
        )

        CellRunner.__init__(
            self,
            net_config=net_config,
            my_name=FQCN.ROOT_SERVER,
        )

    def start(self):
        super().start()
        self.admin.start()

    def stop(self):
        self.admin.stop()
        super().stop()

    def get_spec(self) -> CommandModuleSpec:
        return CommandModuleSpec(
            name="sys",
            cmd_specs=[
                CommandSpec(
                    name="cells",
                    description="get system cells info",
                    usage="cells",
                    handler_func=self._cmd_cells,
                    visible=True,
                ),
                CommandSpec(
                    name="stop",
                    description="stop system",
                    usage="stop",
                    handler_func=self._cmd_stop,
                    visible=True,
                )])

    def _cmd_cells(self, conn: Connection, args: [str]):
        cell_fqcns = self.request_cells_info()
        for c in cell_fqcns:
            conn.append_string(c)

    def _cmd_stop(self, conn: Connection, args: [str]):
        self.stop()
