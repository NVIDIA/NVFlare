import argparse
import json
import os
import shutil
import time
import traceback

import nvflare.fuel.hci.file_transfer_defs as ftd
from nvflare.fuel.hci.base64_utils import b64str_to_bytes, bytes_to_b64str
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import ConfirmMethod
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.fuel.hci.security import hash_password
from nvflare.fuel.hci.server.authz import AuthzFilter, PreAuthzReturnCode
from nvflare.fuel.hci.server.builtin import new_command_register_with_builtin_module
from nvflare.fuel.hci.server.hci import AdminServer
from nvflare.fuel.hci.server.login import LoginModule, SessionManager, SimpleAuthenticator
from nvflare.fuel.hci.zip_utils import unzip_all_from_bytes, zip_directory_to_bytes
from nvflare.fuel.sec.authz import AuthorizationService, Authorizer

CMD_CATS = {
    "add": "math",
    "sub": "math",
    "upload_job": "job",
    "download_job": "job",
    "ls": "shell",
    "cat": "shell",
    "tail": "shell",
    "head": "shell",
    "grep": "shell",
    "stop": "op",
}


class MyAuthorizer(Authorizer):
    def __init__(self, site_org, policy_file_path: str):
        policy_config = json.load(open(policy_file_path, "rt"))
        Authorizer.__init__(self, site_org=site_org, right_categories=CMD_CATS)
        self.load_policy(policy_config)


class CmdModule(CommandModule):
    def get_spec(self):
        return CommandModuleSpec(
            name="cmds",
            cmd_specs=[
                CommandSpec(
                    name="add",
                    description="add two numbers",
                    usage="add x y",
                    handler_func=self.handle_add,
                    authz_func=self.authorize_cmd,
                ),
                CommandSpec(
                    name="sub",
                    description="sub two numbers",
                    usage="add x y",
                    handler_func=self.handle_sub,
                    authz_func=self.authorize_cmd,
                ),
                CommandSpec(
                    name="upload_job",
                    description="upload a job folder",
                    usage="upload_job folder",
                    handler_func=self.handle_upload,
                    authz_func=self.authorize_cmd,
                    client_cmd=ftd.UPLOAD_FOLDER_FQN,
                    visible=True,
                ),
                CommandSpec(
                    name="download_job",
                    description="download a job",
                    usage="download_job job_id",
                    handler_func=self.handle_download,
                    authz_func=self.authorize_cmd,
                    visible=True,
                    client_cmd=ftd.DOWNLOAD_FOLDER_FQN,
                ),
                CommandSpec(
                    name="stop",
                    description="stop system",
                    usage="stop",
                    handler_func=self.handle_stop,
                    authz_func=self.authorize_cmd,
                    confirm=ConfirmMethod.AUTH,
                ),
            ],
        )

    def authorize_cmd(self, conn: Connection, args: [str]):
        print("called to authorize cmd {}".format(args[0]))
        return PreAuthzReturnCode.REQUIRE_AUTHZ

    def handle_stop(self, conn: Connection, args: [str]):
        ctx = conn.app_ctx
        ctx["stop"] = True
        conn.append_shutdown("Have a nice day!")

    def handle_add(self, conn: Connection, args: [str]):
        if len(args) != 3:
            conn.append_error("usage: {} x y".format(args[0]))
            return

        try:
            x = float(args[1])
            y = float(args[2])
            z = x + y
            conn.append_string("result: {}".format(z))
        except BaseException as ex:
            conn.append_error("bad input: {}".format(ex))

    def handle_sub(self, conn: Connection, args: [str]):
        if len(args) != 3:
            conn.append_error("usage: {} x y".format(args[0]))
            return

        try:
            x = float(args[1])
            y = float(args[2])
            z = x - y
            conn.append_string("result: {}".format(z))
        except BaseException as ex:
            conn.append_error("bad input: {}".format(ex))

    def handle_upload(self, conn: Connection, args: [str]):
        folder_name = args[1]
        zip_b64str = args[2]
        upload_dir = conn.get_prop("upload_dir")
        folder_path = os.path.join(upload_dir, folder_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        data_bytes = b64str_to_bytes(zip_b64str)
        unzip_all_from_bytes(data_bytes, upload_dir)
        conn.append_string("Created job folder {}".format(folder_path))

    def handle_download(self, conn: Connection, args: [str]):
        job_id = args[1]
        download_dir = conn.get_prop("download_dir")
        try:
            data = zip_directory_to_bytes(download_dir, job_id)
            b64str = bytes_to_b64str(data)
            conn.append_string(b64str)
        except FileNotFoundError:
            conn.append_error("No record for job '{}'".format(job_id))
        except BaseException:
            traceback.print_exc()
            conn.append_error("Exception occurred during attempt to zip data to send for job: {}".format(job_id))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, help="port number", required=True)
    parser.add_argument("--ssl", "-s", action="store_true", help="ssl or not")

    args = parser.parse_args()

    ctx = {"stop": False}
    cmd_reg = new_command_register_with_builtin_module(app_ctx=ctx)
    authenticator = SimpleAuthenticator(users={"admin": hash_password("admin")})
    session_mgr = SessionManager(idle_timeout=1000)
    login_module = LoginModule(authenticator, session_mgr)
    cmd_reg.register_module(login_module)
    cmd_reg.register_module(session_mgr)
    cmd_reg.register_module(CmdModule())

    cmd_reg.add_filter(login_module)
    cmd_reg.add_filter(AuthzFilter())

    AuthorizationService.initialize(
        MyAuthorizer(
            site_org="nv", policy_file_path="/Users/yanc/flarehub/NVFlare/nvflare/fuel/hci/yan/authz_policy.json"
        )
    )

    p = args.port
    ca_cert = "/Users/yanc/certs/rootCA.pem"
    if not args.ssl:
        ca_cert = None

    server = AdminServer(
        cmd_reg,
        "localhost",
        p,
        ca_cert=ca_cert,
        server_cert="/Users/yanc/certs/server.crt",
        server_key="/Users/yanc/certs/server.key",
        accepted_client_cns=["admin"],
        extra_conn_props={"upload_dir": "/Users/yanc/dlmed/server_up", "download_dir": "/Users/yanc/dlmed/server_down"},
    )

    server.start()
    if args.ssl:
        print(f"Started Admin Server on Port {p} with SSL")
    else:
        print(f"Started Admin Server on Port {p} without SSL")

    while not ctx["stop"]:
        time.sleep(0.5)

    server.stop()
    session_mgr.shutdown()
    print("Server Stopped")


if __name__ == "__main__":
    main()
