# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import time
from typing import List

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import AdminCommandNames, MachineStatus
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.private.defs import ClientStatusKey, RequestHeader, TrainingTopic
from nvflare.private.fed.server.admin import new_message
from nvflare.private.fed.server.server_engine_internal_spec import ServerEngineInternalSpec
from nvflare.security.security import Action, FLAuthzContext

from .app_authz import AppAuthzService
from .cmd_utils import CommandUtil


class TrainingCommandModule(CommandModule, CommandUtil):

    APP_STAGING_PATH = "app_staging_path"

    def __init__(self):
        """A class for training commands."""
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_spec(self):
        return CommandModuleSpec(
            name="training",
            cmd_specs=[
                CommandSpec(
                    name=AdminCommandNames.SET_RUN_NUMBER,
                    description="set the run number",
                    usage="set_run_number number",
                    handler_func=self.set_run_number,
                    authz_func=self.authorize_set_run_number,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.DELETE_RUN_NUMBER,
                    description="delete a run",
                    usage="delete_run_number number",
                    handler_func=self.delete_run_number,
                    authz_func=self.authorize_set_run_number,
                    visible=True,
                    confirm="auth",
                ),
                CommandSpec(
                    name=AdminCommandNames.DEPLOY_APP,
                    description="deploy FL app to client/server",
                    usage="deploy_app server|client <client-name>|all",
                    handler_func=self.deploy_app,
                    authz_func=self.authorize_deploy_app,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.START_APP,
                    description="start the FL app",
                    usage="start_app server|client|all",
                    handler_func=self.start_app,
                    authz_func=self.authorize_train,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.CHECK_STATUS,
                    description="check status of the FL server/client",
                    usage="check_status server|client",
                    handler_func=self.check_status,
                    authz_func=self.authorize_view,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.ABORT,
                    description="abort the FL app",
                    usage="abort server|client|all",
                    handler_func=self.abort_app,
                    authz_func=self.authorize_train,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.ABORT_TASK,
                    description="abort the client current task execution",
                    usage="abort_task <client-name>",
                    handler_func=self.abort_task,
                    authz_func=self.authorize_abort_client,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.REMOVE_CLIENT,
                    description="remove a FL client",
                    usage="remove_client <client-name>",
                    handler_func=self.remove_client,
                    authz_func=self.authorize_remove_client,
                    visible=True,
                    confirm="auth",
                ),
                CommandSpec(
                    name=AdminCommandNames.SHUTDOWN,
                    description="shutdown the FL server/client",
                    usage="shutdown server|client|all",
                    handler_func=self.shutdown,
                    authz_func=self.authorize_operate,
                    visible=True,
                    confirm="auth",
                ),
                CommandSpec(
                    name=AdminCommandNames.RESTART,
                    description="restart the FL server/client",
                    usage="restart server|client|all",
                    handler_func=self.restart,
                    authz_func=self.authorize_operate,
                    visible=True,
                    confirm="auth",
                ),
                CommandSpec(
                    name=AdminCommandNames.SET_TIMEOUT,
                    description="set the admin commands timeout",
                    usage="set_timeout seconds ",
                    handler_func=self.set_timeout,
                    authz_func=self.authorize_set_timeout,
                    visible=True,
                ),
            ],
        )

    # Set Run Number
    def authorize_set_run_number(self, conn: Connection, args: List[str]):
        if len(args) < 2:
            conn.append_error("syntax error: missing run number")
            return False, None

        try:
            num = int(args[1])
        except ValueError:
            conn.append_error("run number must be an integer.")
            return False, None

        if num < 1:
            conn.append_error("run number must be > 0.")
            return False, None

        return True, FLAuthzContext.new_authz_context(site_names=[self.SITE_SERVER], actions=[Action.TRAIN])

    def set_run_number(self, conn: Connection, args: List[str]):
        num = int(args[1])
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))
        conn.append_string(engine.set_run_number(num))
        self._set_run_number_clients(conn, num)

    def _set_run_number_clients(self, conn: Connection, run_number) -> bool:
        engine = conn.app_ctx
        clients = engine.get_clients()
        if clients:
            valid_tokens = []
            for c in clients:
                valid_tokens.append(c.token)
            conn.set_prop(self.TARGET_CLIENT_TOKENS, valid_tokens)

            message = new_message(conn, topic=TrainingTopic.SET_RUN_NUMBER, body="")
            message.set_header(RequestHeader.RUN_NUM, str(run_number))
            replies = self.send_request_to_clients(conn, message)
            self.process_replies_to_table(conn, replies)
            return True

    def delete_run_number(self, conn: Connection, args: List[str]):
        num = int(args[1])
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        engine_info = engine.get_engine_info()

        if engine_info.status == MachineStatus.STARTED or engine_info.status == MachineStatus.STARTING:
            run_number = engine.get_run_number()
            if run_number == num:
                conn.append_error("Current running run_number can not be deleted.")
                return

        err = engine.delete_run_number(num)
        if err:
            conn.append_error(err)
            return

        # ask clients to delete this RUN
        message = new_message(conn, topic=TrainingTopic.DELETE_RUN, body="")
        message.set_header(RequestHeader.RUN_NUM, str(num))
        clients = engine.get_clients()
        if clients:
            conn.set_prop(self.TARGET_CLIENT_TOKENS, [x.token for x in clients])
            replies = self.send_request_to_clients(conn, message)
            self.process_replies_to_table(conn, replies)

        conn.append_success("")

    # Deploy
    def authorize_deploy_app(self, conn: Connection, args: List[str]):
        if len(args) < 3:
            conn.append_error("syntax error: missing target")
            return False, None

        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))
        if engine.get_run_number() < 0:
            conn.append_error("Please set a run number.")
            return False, None

        err = self.validate_command_targets(conn, args[2:])
        if err:
            conn.append_error(err)
            return False, None

        app_name = args[1]
        app_staging_path = engine.get_staging_path_of_app(app_name)
        if not app_staging_path:
            conn.append_error("App {} does not exist. Please upload it first".format(app_name))
            return False, None

        conn.set_prop(self.APP_STAGING_PATH, app_staging_path)
        target_type = args[2]

        if target_type == self.TARGET_TYPE_SERVER:
            sites = [self.SITE_SERVER]
        else:
            sites = []
            client_names = conn.get_prop(self.TARGET_CLIENT_NAMES)
            if client_names:
                sites.extend(client_names)

            if target_type == self.TARGET_TYPE_ALL:
                sites.append(self.SITE_SERVER)

        err, authz_ctx = AppAuthzService.authorize_deploy(app_staging_path, sites)
        if err:
            conn.append_error(err)
            return False, None
        else:

            return True, authz_ctx

    def _deploy_to_clients(self, conn: Connection, app_name, app_staging_path) -> bool:
        # return True if successful
        engine = conn.app_ctx
        client_names = conn.get_prop(self.TARGET_CLIENT_NAMES)
        # for client_name in client_names:
        #     err = engine.prepare_deploy_app_to_client(app_name, app_staging_path, client_name)
        #     if err:
        #         conn.append_error(err)
        #         return False

        err, app_data = engine.get_app_data(app_name)
        if err:
            conn.append_error(err)
            return False

        message = new_message(conn, topic=TrainingTopic.DEPLOY, body=app_data)
        message.set_header(RequestHeader.RUN_NUM, str(engine.get_run_number()))
        message.set_header(RequestHeader.APP_NAME, app_name)
        replies = self.send_request_to_clients(conn, message)
        self.process_replies_to_table(conn, replies)
        return True

    def _deploy_to_server(self, conn, app_name, app_staging_path) -> bool:
        # return True if successful
        engine = conn.app_ctx
        err = engine.deploy_app_to_server(app_name, app_staging_path)
        if not err:
            conn.append_string('deployed app "{}" to Server'.format(app_name))
            return True
        else:
            conn.append_error(err)
            return False

    def deploy_app(self, conn: Connection, args: List[str]):
        app_name = args[1]

        target_type = conn.get_prop(self.TARGET_TYPE)
        app_staging_path = conn.get_prop(self.APP_STAGING_PATH)
        if target_type == self.TARGET_TYPE_SERVER:
            if not self._deploy_to_server(conn, app_name, app_staging_path):
                return
        elif target_type == self.TARGET_TYPE_CLIENT:
            if not self._deploy_to_clients(conn, app_name, app_staging_path):
                return
        else:
            # all
            success = self._deploy_to_server(conn, app_name, app_staging_path)
            if success:
                client_names = conn.get_prop(self.TARGET_CLIENT_NAMES, None)
                if client_names:
                    if not self._deploy_to_clients(conn, app_name, app_staging_path):
                        return
            else:
                return
        conn.append_success("")

    # Start App
    def _start_app_on_server(self, conn: Connection) -> bool:
        engine = conn.app_ctx
        err = engine.start_app_on_server()
        if err:
            conn.append_error(err)
            return False
        else:
            conn.append_string("Server app is starting....")
            return True

    def _start_app_on_clients(self, conn: Connection) -> bool:
        engine = conn.app_ctx
        err = engine.check_app_start_readiness()
        if err:
            conn.append_error(err)
            return False

        run_info = engine.get_run_info()
        message = new_message(conn, topic=TrainingTopic.START, body="")
        message.set_header(RequestHeader.RUN_NUM, str(run_info.run_number))
        replies = self.send_request_to_clients(conn, message)
        self.process_replies_to_table(conn, replies)
        return True

    def start_app(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        target_type = args[1]
        if target_type == self.TARGET_TYPE_SERVER:
            if not self._start_app_on_server(conn):
                return
        elif target_type == self.TARGET_TYPE_CLIENT:
            if not self._start_app_on_clients(conn):
                return
        else:
            # all
            success = self._start_app_on_server(conn)

            if success:
                # engine_info = None
                # start = time.time()
                # # Wait for the server App to start properly
                # while engine_info is None or engine_info.status != MachineStatus.STARTED:
                #     time.sleep(0.3)
                #     engine_info = engine.get_engine_info()
                #     if time.time() - start > 60.0:
                #         conn.append_error("Could not start the server app")
                #         return
                #
                client_names = conn.get_prop(self.TARGET_CLIENT_NAMES, None)
                if client_names:
                    if not self._start_app_on_clients(conn):
                        return
        conn.append_success("")

    # Abort App
    def _abort_clients(self, conn, clients: List[str]) -> bool:
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        err = engine.abort_app_on_clients(clients)
        if err:
            conn.append_error(err)
            return False

        run_info = engine.get_run_info()
        message = new_message(conn, topic=TrainingTopic.ABORT, body="")
        if run_info:
            message.set_header(RequestHeader.RUN_NUM, str(run_info.run_number))

        # conn.set_prop(self.TARGET_CLIENT_NAMES, client_names)
        replies = self.send_request_to_clients(conn, message)
        self.process_replies_to_table(conn, replies)
        return True

    def abort_app(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        target_type = args[1]
        if target_type == self.TARGET_TYPE_SERVER or target_type == self.TARGET_TYPE_ALL:
            conn.append_string("Trying to abort all clients before abort server ...")
            clients = engine.get_clients()
            if clients:
                tokens = [c.token for c in clients]
                conn.set_prop(
                    self.TARGET_CLIENT_TOKENS, tokens
                )  # need this because not set in validate_command_targets when target_type == self.TARGET_TYPE_SERVER
                if not self._abort_clients(conn, clients=[c.token for c in clients]):
                    return
            err = engine.abort_app_on_server()
            if err:
                conn.append_error(err)
                return
            conn.append_string("Abort signal has been sent to the server app.")
        elif target_type == self.TARGET_TYPE_CLIENT:
            clients = conn.get_prop(self.TARGET_CLIENT_TOKENS)
            if not clients:
                conn.append_string("No clients to abort")
                return
            if not self._abort_clients(conn, clients):
                return
        conn.append_success("")

    def abort_task(self, conn, clients: List[str]) -> str:
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        err = engine.abort_app_on_clients(clients)
        if err:
            conn.append_error(err)
            return ""

        run_info = engine.get_run_info()
        message = new_message(conn, topic=TrainingTopic.ABORT_TASK, body="")
        if run_info:
            message.set_header(RequestHeader.RUN_NUM, str(run_info.run_number))

        # conn.set_prop(self.TARGET_CLIENT_NAMES, client_names)
        replies = self.send_request_to_clients(conn, message)
        return self.process_replies_to_table(conn, replies)

    # Shutdown
    def _shutdown_app_on_server(self, conn: Connection) -> bool:
        engine = conn.app_ctx
        err = engine.shutdown_server()
        if err:
            conn.append_error(err)
            return False
        else:
            conn.append_string("FL app has been shutdown.")
            conn.append_shutdown("Bye bye")
            return True

    def _shutdown_app_on_clients(self, conn: Connection) -> bool:
        engine = conn.app_ctx
        message = new_message(conn, topic=TrainingTopic.SHUTDOWN, body="")
        clients = conn.get_prop(self.TARGET_CLIENT_TOKENS, None)
        if not clients:
            conn.append_error("no clients to shutdown")
            return False

        replies = self.send_request_to_clients(conn, message)
        self.process_replies_to_table(conn, replies)

        err = engine.remove_clients(clients)
        if err:
            conn.append_error(err)
            return False
        return True

    def shutdown(self, conn: Connection, args: List[str]):
        target_type = args[1]
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        if target_type == self.TARGET_TYPE_SERVER:
            if engine.get_clients():
                conn.append_error("There are still active clients. Shutdown all clients first.")
                return
            if not self._shutdown_app_on_server(conn):
                return
        elif target_type == self.TARGET_TYPE_CLIENT:
            if not self._shutdown_app_on_clients(conn):
                return
        else:
            # all
            if engine.get_clients():
                conn.append_string("Trying to shutdown clients before server...")
                success = self._shutdown_app_on_clients(conn)
                if success:
                    if not self._shutdown_app_on_server(conn):
                        return
            else:
                if not self._shutdown_app_on_server(conn):
                    return
        conn.append_success("")

    # Remove Clients
    def authorize_remove_client(self, conn: Connection, args: List[str]):
        if len(args) < 2:
            conn.append_error("syntax error: missing site names")
            return False, None

        auth_args = [args[0], self.TARGET_TYPE_CLIENT]
        auth_args.extend(args[1:])
        return self.authorize_operate(conn, auth_args)

    def authorize_abort_client(self, conn: Connection, args: List[str]):
        auth_args = [args[0], self.TARGET_TYPE_CLIENT]
        auth_args.extend(args[1:])
        return self.authorize_operate(conn, auth_args)

    def remove_client(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))
        clients = conn.get_prop(self.TARGET_CLIENT_TOKENS)
        err = engine.remove_clients(clients)
        if err:
            conn.append_error(err)
            return
        conn.append_success("")

    # Restart
    def _restart_clients(self, conn, clients) -> str:
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))
        engine.remove_clients(clients)
        message = new_message(conn, topic=TrainingTopic.RESTART, body="")
        replies = self.send_request_to_clients(conn, message)
        return self._process_replies_to_string(conn, replies)

    def restart(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        target_type = args[1]
        if target_type == self.TARGET_TYPE_SERVER or target_type == self.TARGET_TYPE_ALL:

            clients = engine.get_clients()
            if clients:
                conn.append_string("Trying to restart all clients before restarting server...")
                tokens = [c.token for c in clients]
                conn.set_prop(
                    self.TARGET_CLIENT_TOKENS, tokens
                )  # need this because not set in validate_command_targets when target_type == self.TARGET_TYPE_SERVER
                response = self._restart_clients(conn, tokens)
                conn.append_string(response)
                # check with Isaac - no need to wait!
                # time.sleep(5)

            err = engine.restart_server()
            if err:
                conn.append_error(err)
            else:
                conn.append_string("Server scheduled for restart")
        elif target_type == self.TARGET_TYPE_CLIENT:
            clients = conn.get_prop(self.TARGET_CLIENT_TOKENS)
            if not clients:
                conn.append_error("no clients available")
                return
            else:
                response = self._restart_clients(conn, clients)
                conn.append_string(response)
        conn.append_success("")

    # Set Timeout
    def authorize_set_timeout(self, conn: Connection, args: List[str]):
        if len(args) != 2:
            conn.append_error("syntax error: missing timeout")
            return False, None

        try:
            num = float(args[1])
        except ValueError:
            conn.append_error("must provide the timeout value in seconds")
            return False, None

        if num <= 0:
            conn.append_error("timeout must be > 0")
            return False, None

        return True, FLAuthzContext.new_authz_context(site_names=[self.SITE_SERVER], actions=[Action.TRAIN])

    def set_timeout(self, conn: Connection, args: List[str]):
        timeout = float(args[1])
        server = conn.server
        server.timeout = timeout
        conn.append_string("admin command timeout has been set to: {}".format(timeout))
        conn.append_success("")

    # Check status
    def check_status(self, conn: Connection, args: List[str]):
        # TODO:: Need more discussion on what status to be shown
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))
        dest = args[1]
        if dest == self.TARGET_TYPE_SERVER:
            engine_info = engine.get_engine_info()
            conn.append_string(f"FL_app name: {engine_info.app_name}")
            conn.append_string(f"Engine status: {engine_info.status.value}")
            run_info = engine.get_run_info()
            if engine.get_run_number() < 0:
                conn.append_string("Run number has not been set.")
            else:
                conn.append_string(f"Current run number: {engine.get_run_number()}")

            clients = engine.get_clients()
            conn.append_string("Registered clients: {} ".format(len(clients)))

            if clients:
                table = conn.append_table(["Client", "Token", "Last Connect Time"])
                for c in clients:
                    if not isinstance(c, Client):
                        raise TypeError("c must be Client but got {}".format(type(c)))
                    table.add_row([c.name, str(c.token), time.asctime(time.localtime(c.last_connect_time))])
        elif dest == self.TARGET_TYPE_CLIENT:
            message = new_message(conn, topic=TrainingTopic.CHECK_STATUS, body="")
            replies = self.send_request_to_clients(conn, message)
            self._process_status_replies(conn, replies)
        else:
            conn.append_error("invalid target type {}. Usage: check_status server|client ...".format(dest))

    def _process_status_replies(self, conn, replies):
        if not replies:
            conn.append_error("no responses from clients")
            return

        engine = conn.app_ctx
        table = conn.append_table(["client", "app_name", "run_number", "status"])
        for r in replies:
            run_num = "?"
            status = "?"
            app_name = "?"
            client_name = engine.get_client_name_from_token(r.client_token)

            if r.reply:
                try:
                    body = json.loads(r.reply.body)
                    if r.reply and isinstance(body, dict):
                        app_name = body.get(ClientStatusKey.APP_NAME, "?")
                        run_num = body.get(ClientStatusKey.RUN_NUM, "?")
                        status = body.get(ClientStatusKey.STATUS, "?")
                except BaseException:
                    self.logger.error("Bad reply from client")

                table.add_row([client_name, app_name, run_num, status])
            else:
                table.add_row([client_name, app_name, run_num, "No Reply"])
