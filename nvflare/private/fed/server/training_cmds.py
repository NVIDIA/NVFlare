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
from nvflare.apis.fl_constant import AdminCommandNames, WorkspaceConstants
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.reg import CommandModule, CommandModuleSpec, CommandSpec
from nvflare.private.defs import ClientStatusKey, RequestHeader, TrainingTopic
from nvflare.private.fed.server.admin import new_message
from nvflare.private.fed.server.server_engine_internal_spec import ServerEngineInternalSpec
from nvflare.security.security import Action, FLAuthzContext

from .app_authz import AppAuthzService
from .cmd_utils import CommandUtil
from .server_engine import ServerEngine


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
                    name=AdminCommandNames.DELETE_WORKSPACE,
                    description="delete the workspace of a job",
                    usage="delete_workspace job_id",
                    handler_func=self.delete_job_id,
                    authz_func=self.authorize_set_job_id,
                    visible=False,
                    confirm="auth",
                ),
                CommandSpec(
                    name=AdminCommandNames.DEPLOY_APP,
                    description="deploy FL app to client/server",
                    usage="deploy_app job_id app server|client <client-name>|all",
                    handler_func=self.deploy_app,
                    authz_func=self.authorize_deploy_app,
                    visible=False,
                ),
                CommandSpec(
                    name=AdminCommandNames.START_APP,
                    description="start the FL app",
                    usage="start_app job_id server|client|all",
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
                    usage="abort job_id server|client|all",
                    handler_func=self.abort_app,
                    authz_func=self.authorize_train,
                    visible=False,
                ),
                CommandSpec(
                    name=AdminCommandNames.ABORT_TASK,
                    description="abort the client current task execution",
                    usage="abort_task job_id <client-name>",
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

    def authorize_set_job_id(self, conn: Connection, args: List[str]):
        if len(args) < 2:
            conn.append_error("syntax error: missing job id")
            return False, None

        return True, FLAuthzContext.new_authz_context(site_names=[self.SITE_SERVER], actions=[Action.TRAIN])

    def _set_job_id_clients(self, conn: Connection, job_id) -> bool:
        engine = conn.app_ctx
        clients = engine.get_clients()
        if clients:
            valid_tokens = []
            for c in clients:
                valid_tokens.append(c.token)
            conn.set_prop(self.TARGET_CLIENT_TOKENS, valid_tokens)

            message = new_message(conn, topic=TrainingTopic.SET_JOB_ID, body="")
            message.set_header(RequestHeader.JOB_ID, str(job_id))
            replies = self.send_request_to_clients(conn, message)
            self.process_replies_to_table(conn, replies)
            return True

    def delete_job_id(self, conn: Connection, args: List[str]):
        job_id = args[1]
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngine):
            raise TypeError("engine must be ServerEngine but got {}".format(type(engine)))

        if job_id in engine.run_processes.keys():
            conn.append_error(f"Current running run_{job_id} can not be deleted.")
            return

        err = engine.delete_job_id(job_id)
        if err:
            conn.append_error(err)
            return

        # ask clients to delete this RUN
        message = new_message(conn, topic=TrainingTopic.DELETE_RUN, body="")
        message.set_header(RequestHeader.JOB_ID, str(job_id))
        clients = engine.get_clients()
        if clients:
            conn.set_prop(self.TARGET_CLIENT_TOKENS, [x.token for x in clients])
            replies = self.send_request_to_clients(conn, message)
            self.process_replies_to_table(conn, replies)

        conn.append_success("")

    # Deploy
    def authorize_deploy_app(self, conn: Connection, args: List[str]):
        if len(args) < 4:
            conn.append_error("syntax error: missing job_id and target")
            return False, None

        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        err = self.validate_command_targets(conn, args[3:])
        if err:
            conn.append_error(err)
            return False, None

        run_destination = args[1].lower()
        if not run_destination.startswith(WorkspaceConstants.WORKSPACE_PREFIX):
            conn.append_error("syntax error: run_destination must be run_XXX")
            return False, None
        destination = run_destination[len(WorkspaceConstants.WORKSPACE_PREFIX) :]
        conn.set_prop(self.JOB_ID, destination)

        app_name = args[2]
        app_staging_path = engine.get_staging_path_of_app(app_name)
        if not app_staging_path:
            conn.append_error("App {} does not exist. Please upload it first".format(app_name))
            return False, None

        conn.set_prop(self.APP_STAGING_PATH, app_staging_path)
        target_type = args[3]

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

    def _deploy_to_clients(self, conn: Connection, app_name, job_id) -> bool:
        # return True if successful
        engine = conn.app_ctx
        err, app_data = engine.get_app_data(app_name)
        if err:
            conn.append_error(err)
            return False

        message = new_message(conn, topic=TrainingTopic.DEPLOY, body=app_data)
        message.set_header(RequestHeader.JOB_ID, str(job_id))
        message.set_header(RequestHeader.APP_NAME, app_name)
        replies = self.send_request_to_clients(conn, message)
        self.process_replies_to_table(conn, replies)
        return True

    def _deploy_to_server(self, conn, job_id, app_name, app_staging_path) -> bool:
        # return True if successful
        engine = conn.app_ctx
        err = engine.deploy_app_to_server(job_id, app_name, app_staging_path)
        if not err:
            conn.append_string('deployed app "{}" to Server'.format(app_name))
            return True
        else:
            conn.append_error(err)
            return False

    def deploy_app(self, conn: Connection, args: List[str]):
        app_name = args[2]

        job_id = conn.get_prop(self.JOB_ID)
        target_type = conn.get_prop(self.TARGET_TYPE)
        app_staging_path = conn.get_prop(self.APP_STAGING_PATH)
        if target_type == self.TARGET_TYPE_SERVER:
            if not self._deploy_to_server(conn, job_id, app_name, app_staging_path):
                return
        elif target_type == self.TARGET_TYPE_CLIENT:
            if not self._deploy_to_clients(conn, app_name, job_id):
                return
        else:
            # all
            success = self._deploy_to_server(conn, job_id, app_name, app_staging_path)
            if success:
                client_names = conn.get_prop(self.TARGET_CLIENT_NAMES, None)
                if client_names:
                    if not self._deploy_to_clients(conn, app_name, job_id):
                        return
            else:
                return
        conn.append_success("")

    # Start App
    def _start_app_on_server(self, conn: Connection, job_id: str) -> bool:
        engine = conn.app_ctx
        err = engine.start_app_on_server(job_id)
        if err:
            conn.append_error(err)
            return False
        else:
            conn.append_string("Server app is starting....")
            return True

    def _start_app_on_clients(self, conn: Connection, job_id: str) -> bool:
        engine = conn.app_ctx
        err = engine.check_app_start_readiness(job_id)
        if err:
            conn.append_error(err)
            return False

        # run_info = engine.get_run_info()
        message = new_message(conn, topic=TrainingTopic.START, body="")
        # message.set_header(RequestHeader.JOB_ID, str(run_info.job_id))
        message.set_header(RequestHeader.JOB_ID, job_id)
        replies = self.send_request_to_clients(conn, message)
        self.process_replies_to_table(conn, replies)
        return True

    def start_app(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        job_id = conn.get_prop(self.JOB_ID)
        target_type = args[2]
        if target_type == self.TARGET_TYPE_SERVER:
            if not self._start_app_on_server(conn, job_id):
                return
        elif target_type == self.TARGET_TYPE_CLIENT:
            if not self._start_app_on_clients(conn, job_id):
                return
        else:
            # all
            success = self._start_app_on_server(conn, job_id)

            if success:
                client_names = conn.get_prop(self.TARGET_CLIENT_NAMES, None)
                if client_names:
                    if not self._start_app_on_clients(conn, job_id):
                        return
        conn.append_success("")

    # Abort App
    def _abort_clients(self, conn, clients: List[str], job_id) -> bool:
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        err = engine.abort_app_on_clients(clients)
        if err:
            conn.append_error(err)
            return False

        # run_info = engine.get_app_run_info(job_id)
        message = new_message(conn, topic=TrainingTopic.ABORT, body="")
        # if run_info:
        message.set_header(RequestHeader.JOB_ID, str(job_id))

        # conn.set_prop(self.TARGET_CLIENT_NAMES, client_names)
        replies = self.send_request_to_clients(conn, message)
        self.process_replies_to_table(conn, replies)
        return True

    def abort_app(self, conn: Connection, args: List[str]):
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        job_id = conn.get_prop(self.JOB_ID)
        target_type = args[2]
        if target_type == self.TARGET_TYPE_SERVER or target_type == self.TARGET_TYPE_ALL:
            conn.append_string("Trying to abort all clients before abort server ...")
            clients = engine.get_clients()
            if clients:
                tokens = [c.token for c in clients]
                conn.set_prop(
                    self.TARGET_CLIENT_TOKENS, tokens
                )  # need this because not set in validate_command_targets when target_type == self.TARGET_TYPE_SERVER
                if not self._abort_clients(conn, clients=[c.token for c in clients], job_id=job_id):
                    return
            err = engine.abort_app_on_server(job_id)
            if err:
                conn.append_error(err)
                return
            conn.append_string("Abort signal has been sent to the server app.")
        elif target_type == self.TARGET_TYPE_CLIENT:
            clients = conn.get_prop(self.TARGET_CLIENT_TOKENS)
            if not clients:
                conn.append_string("No clients to abort")
                return
            if not self._abort_clients(conn, clients, job_id):
                return
        conn.append_success("")

    def abort_task(self, conn, clients: List[str]) -> str:
        engine = conn.app_ctx
        if not isinstance(engine, ServerEngineInternalSpec):
            raise TypeError("engine must be ServerEngineInternalSpec but got {}".format(type(engine)))

        job_id = conn.get_prop(self.JOB_ID)
        # run_info = engine.get_app_run_info()
        message = new_message(conn, topic=TrainingTopic.ABORT_TASK, body="")
        # if run_info:
        message.set_header(RequestHeader.JOB_ID, str(job_id))

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
        if not isinstance(engine, ServerEngine):
            raise TypeError("engine must be ServerEngine but got {}".format(type(engine)))

        if engine.job_runner.running_jobs:
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
    def authorize_remove_client(self, conn: Connection, args: List[str]):
        if len(args) < 2:
            conn.append_error("syntax error: missing site names")
            return False, None

        auth_args = [args[0], self.TARGET_TYPE_CLIENT]
        auth_args.extend(args[1:])
        return self.authorize_operate(conn, auth_args)

    def authorize_abort_client(self, conn: Connection, args: List[str]):
        if len(args) < 3:
            conn.append_error("syntax error: missing job_id and target")
            return False, None

        run_destination = args[1].lower()
        if not run_destination.startswith(WorkspaceConstants.WORKSPACE_PREFIX):
            conn.append_error("syntax error: run_destination must be run_XXX")
            return False, None
        job_id = run_destination[len(WorkspaceConstants.WORKSPACE_PREFIX) :]
        conn.set_prop(self.JOB_ID, job_id)

        auth_args = [args[0], self.TARGET_TYPE_CLIENT]
        auth_args.extend(args[2:])
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
        dst = args[1]
        if dst == self.TARGET_TYPE_SERVER:
            engine_info = engine.get_engine_info()
            conn.append_string(f"Engine status: {engine_info.status.value}")
            table = conn.append_table(["Job_id", "App Name"])
            for job_id, app_name in engine_info.app_names.items():
                table.add_row([job_id, app_name])

            clients = engine.get_clients()
            conn.append_string("Registered clients: {} ".format(len(clients)))

            if clients:
                table = conn.append_table(["Client", "Token", "Last Connect Time"])
                for c in clients:
                    if not isinstance(c, Client):
                        raise TypeError("c must be Client but got {}".format(type(c)))
                    table.add_row([c.name, str(c.token), time.asctime(time.localtime(c.last_connect_time))])
        elif dst == self.TARGET_TYPE_CLIENT:
            message = new_message(conn, topic=TrainingTopic.CHECK_STATUS, body="")
            replies = self.send_request_to_clients(conn, message)
            self._process_status_replies(conn, replies)
        else:
            conn.append_error("invalid target type {}. Usage: check_status server|client ...".format(dst))

    def _process_status_replies(self, conn, replies):
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
                try:
                    body = json.loads(r.reply.body)
                    if r.reply and isinstance(body, dict):
                        running_jobs = body.get(ClientStatusKey.RUNNING_JOBS)
                        if running_jobs:
                            for job in running_jobs:
                                app_name = job.get(ClientStatusKey.APP_NAME, "?")
                                job_id = job.get(ClientStatusKey.JOB_ID, "?")
                                status = job.get(ClientStatusKey.STATUS, "?")
                                table.add_row([client_name, app_name, job_id, status])
                        else:
                            table.add_row([client_name, app_name, job_id, "No Jobs"])
                except BaseException as ex:
                    self.logger.error(f"Bad reply from client: {ex}")
            else:
                table.add_row([client_name, app_name, job_id, "No Reply"])
