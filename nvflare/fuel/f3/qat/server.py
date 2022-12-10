from nvflare.fuel.hci.server.reg import ServerCommandRegister
from nvflare.fuel.hci.server.hci import AdminServer
from nvflare.fuel.f3.cellnet import Cell


class Server:

    def __init__(
            self,
            net_config_file: str,
            admin_host: str,
            admin_port: int,
    ):
        self.net_config_file = net_config_file
        reg = ServerCommandRegister(app_ctx=self)
        self.admin = AdminServer(
            cmd_reg=reg,
            host=admin_host,
            port=admin_port
        )
        self.cell = Cell(
            fqcn="server",
            root_url="",
            secure=False,
            credentials={}
        )

    def start(self):
        self.cell.start()
        self.admin.start()

    def stop(self):
        self.admin.stop()
        self.cell.stop()
