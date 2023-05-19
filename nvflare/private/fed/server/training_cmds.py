# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from nvflare.apis.fl_constant import AdminCommandNames
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.proto import ConfirmMethod, MetaKey, MetaStatusValue, make_meta
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.private.admin_defs import MsgHeader, ReturnCode
from nvflare.private.defs import ClientStatusKey, ScopeInfoKey, TrainingTopic
from nvflare.private.fed.server.admin import new_message
from nvflare.private.fed.server.server_engine_internal_spec import ServerEngineInternalSpec
from nvflare.private.fed.utils.fed_utils import get_scope_info
from nvflare.security.logging import secure_format_exception

from .cmd_utils import CommandUtil
from .server_engine import ServerEngine


class TrainingCommandModule(CommandModule, CommandUtil):
    def __init__(self):
        """A class for training commands."""
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_spec(self):
        return CommandModuleSpec(
            name="training",
            cmd_specs=[
                CommandSpec(
                    name=AdminCommandNames.CHECK_STATUS,
                    description="check status of the FL server/client",
                    usage="check_status server|client",
                    handler_func=self.check_status,
                    authz_func=self.authorize_server_operation,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.REMOVE_CLIENT,
                    description="remove a FL client",
                    usage="remove_client <client-name>",
                    handler_func=self.remove_client,
                    authz_func=self.authorize_client_operation,
                    visible=True,
                    confirm=ConfirmMethod.AUTH,
                ),
                CommandSpec(
                    name=AdminCommandNames.ADMIN_CHECK_STATUS,
                    description="check status for shutdown system command",
                    usage="admin_check_status",
                    handler_func=self.check_status,
                    authz_func=self.must_be_project_admin,
                    visible=False,
                ),
                CommandSpec(
                    name=AdminCommandNames.SHUTDOWN,
                    description="shutdown the FL server/client",
                    usage="shutdown server|client|all",
                    handler_func=self.shutdown,
                    authz_func=self.authorize_server_operation,
                    visible=True,
                    confirm=ConfirmMethod.AUTH,
                ),
                CommandSpec(
                    name=AdminCommandNames.RESTART,
                    description="restart the FL server/client",
                    usage="restart server|client|all",
                    handler_func=self.restart,
                    authz_func=self.authorize_server_operation,
                    visible=True,
                    confirm=ConfirmMethod.AUTH,
                ),
                CommandSpec(
                    name=AdminCommandNames.SET_TIMEOUT,
                    description="set the admin commands timeout",
                    usage="set_timeout seconds ",
                    handler_func=self.set_timeout,
                    authz_func=self.command_authz_required,
                    visible=True,
                ),
                CommandSpec(
                    name=AdminCommandNames.SHOW_SCOPES,
                    description="show configured scope names on server/client",
                    usage="show_scopes server|client|all ...",
                    handler_func=self.show_scopes,
                    authz_func=self.authorize_server_operation,
                    visible=True,
                ),
            ],
        )

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
        message = new_message(conn, topic=TrainingTopic.SHUTDOWN, body="", require_authz=True)
        clients = conn.get_prop(self.TARGET_CLIENT_TOKENS, None)
        if not clients:
            conn.append_error("no clients to shutdown")
            return False

        replies = self.send_request_to_clients(conn, message)
        self.process_replies_to_table(conn, replies)

        clients_to_be_removed = set(clients)
        for r in replies:
            if r.reply and r.reply.get_header(MsgHeader.RETURN_CODE) == ReturnCode.ERROR:
                clients_to_be_removed.remove(r.client_token)

        result = True
        if clients_to_be_removed != set(clients):
            # means some clients can not be shutdown
            result = False

        return result

    def shutdown(self, conn: Connection, args: List[str]):
        target_type = args[1]
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngine):
            raise TypeError("engine must be ServerEngine but got {}".format(type(engine)))

        for _, job in engine.job_runner.running_jobs.items():
            if not job.run_aborted:
                conn.append_error("There are still jobs running. Please let them finish or abort_job before shutdown.")
                return

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
        message = new_message(conn, topic=TrainingTopic.RESTART, body="", require_authz=True)
        replies = self.send_request_to_clients(conn, message)
        # engine.remove_clients(clients)
        return self._process_replies_to_string(conn, replies)

    def restart(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngine):
            raise TypeError("engine must be ServerEngine but got {}".format(type(engine)))

        if engine.job_runner.running_jobs:
            conn.append_error("There are still jobs running. Please let them finish or abort_job before restart.")
            return

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
        dst = args[1]

        if dst in [self.TARGET_TYPE_SERVER, self.TARGET_TYPE_ALL]:
            engine_info = engine.get_engine_info()
            conn.append_string(
                f"Engine status: {engine_info.status.value}",
                meta=make_meta(
                    MetaStatusValue.OK,
                    extra={
                        MetaKey.SERVER_STATUS: engine_info.status.value,
                        MetaKey.SERVER_START_TIME: engine_info.start_time,
                    },
                ),
            )
            table = conn.append_table(["job_id", "app name"], name=MetaKey.JOBS)
            for job_id, app_name in engine_info.app_names.items():
                table.add_row([job_id, app_name], meta={MetaKey.APP_NAME: app_name, MetaKey.JOB_ID: job_id})

            clients = engine.get_clients()
            conn.append_string("Registered clients: {} ".format(len(clients)))

            if clients:
                table = conn.append_table(["client", "token", "last connect time"], name=MetaKey.CLIENTS)
                for c in clients:
                    if not isinstance(c, Client):
                        raise TypeError("c must be Client but got {}".format(type(c)))
                    table.add_row(
                        [c.name, str(c.token), time.asctime(time.localtime(c.last_connect_time))],
                        meta={MetaKey.CLIENT_NAME: c.name, MetaKey.CLIENT_LAST_CONNECT_TIME: c.last_connect_time},
                    )

        if dst in [self.TARGET_TYPE_CLIENT, self.TARGET_TYPE_ALL]:
            message = new_message(conn, topic=TrainingTopic.CHECK_STATUS, body="", require_authz=True)
            replies = self.send_request_to_clients(conn, message)
            self._process_client_status_replies(conn, replies)

        if dst not in [self.TARGET_TYPE_ALL, self.TARGET_TYPE_CLIENT, self.TARGET_TYPE_SERVER]:
            conn.append_error(
                f"invalid target type {dst}. Usage: check_status server|client ...",
                meta=make_meta(MetaStatusValue.SYNTAX_ERROR, f"invalid target type {dst}"),
            )

    def _process_client_status_replies(self, conn, replies):
        if not replies:
            conn.append_error("no responses from clients")
            return

        engine = conn.app_ctx
        table = conn.append_table(["client", "app_name", "job_id", "status"])
        for r in replies:
            job_id = "?"
            app_name = "?"
            client_name = engine.get_client_name_from_token(r.client_token)

            if r.reply:
                if r.reply.get_header(MsgHeader.RETURN_CODE) == ReturnCode.ERROR:
                    table.add_row([client_name, app_name, job_id, r.reply.body])
                else:
                    try:
                        body = json.loads(r.reply.body)
                        if isinstance(body, dict):
                            running_jobs = body.get(ClientStatusKey.RUNNING_JOBS)
                            if running_jobs:
                                for job in running_jobs:
                                    app_name = job.get(ClientStatusKey.APP_NAME, "?")
                                    job_id = job.get(ClientStatusKey.JOB_ID, "?")
                                    status = job.get(ClientStatusKey.STATUS, "?")
                                    table.add_row([client_name, app_name, job_id, status])
                            else:
                                table.add_row([client_name, app_name, job_id, "No Jobs"])
                    except Exception as e:
                        self.logger.error(f"Bad reply from client: {secure_format_exception(e)}")
            else:
                table.add_row([client_name, app_name, job_id, "No Reply"])

    def _add_scope_info(self, table, site_name, scope_names: List[str], default_scope: str):
        if not scope_names:
            names = ""
        else:
            names = ", ".join(scope_names)
        table.add_row([site_name, names, default_scope])

    def _process_scope_replies(self, table, conn, replies):
        if not replies:
            conn.append_error("no responses from clients")
            return

        engine = conn.app_ctx
        for r in replies:
            client_name = engine.get_client_name_from_token(r.client_token)

            if r.reply:
                if r.reply.get_header(MsgHeader.RETURN_CODE) == ReturnCode.ERROR:
                    self._add_scope_info(table, client_name, r.reply.body, "")
                else:
                    try:
                        body = json.loads(r.reply.body)
                        if isinstance(body, dict):
                            scope_names = body.get(ScopeInfoKey.SCOPE_NAMES)
                            default_scope = body.get(ScopeInfoKey.DEFAULT_SCOPE)
                            self._add_scope_info(table, client_name, scope_names, default_scope)
                        else:
                            conn.append_error(
                                f"bad response from client {client_name}: expect dict but got {type(body)}"
                            )
                    except Exception as e:
                        self.logger.error(f"Bad reply from client: {secure_format_exception(e)}")
                        conn.append_error(f"bad response from client {client_name}: {secure_format_exception(e)}")
            else:
                self._add_scope_info(table, client_name, [], "no reply")

    def show_scopes(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        dst = args[1]
        table = conn.append_table(["site", "scopes", "default"])

        if dst in [self.TARGET_TYPE_SERVER, self.TARGET_TYPE_ALL]:
            # get the server's scope info
            scope_names, default_scope_name = get_scope_info()
            self._add_scope_info(table, "server", scope_names, default_scope_name)

        if dst in [self.TARGET_TYPE_CLIENT, self.TARGET_TYPE_ALL]:
            message = new_message(conn, topic=TrainingTopic.GET_SCOPES, body="", require_authz=True)
            replies = self.send_request_to_clients(conn, message)
            self._process_scope_replies(table, conn, replies)
