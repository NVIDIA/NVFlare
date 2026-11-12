# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import http.client
import json
import os
import shlex
import sys
import threading
import time
from typing import Optional

from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.app_common.metrics_exchange.metrics_sender import ANALYTICS_BOOTSTRAP_ENV, ANALYTICS_BOOTSTRAP_FILE
from nvflare.app_common.tie.applet import Applet
from nvflare.app_common.tie.cli_applet import CLIApplet
from nvflare.app_common.tie.defs import Constant as TieConstant
from nvflare.app_common.tie.process_mgr import CommandDescriptor, ProcessManager, StopMethod
from nvflare.app_opt.flower.defs import Constant
from nvflare.app_opt.flower.http_transport import split_address
from nvflare.app_opt.flower.process import run_command, start_process
from nvflare.security.logging import secure_format_exception

from .path_utils import validate_flower_app_path, validate_flower_app_path_no_symlinks

# Flower CLI executable names
FLOWER_SUPERLINK = "flower-superlink"
FLOWER_SUPERNODE = "flower-supernode"
FLOWER_CLI = "flwr"
FLOWER_CONFIG_FILE = "config.toml"
FLOWER_SUPERLINK_CONNECTION = "nvflare"


def get_partition_id(fl_ctx: FLContext):
    """Get the partition id for the current client based on the sorted list of all client names."""
    engine = fl_ctx.get_engine()

    all_client_names = sorted([client.name for client in engine.get_clients()])

    for id, client_name in enumerate(all_client_names):
        if client_name == fl_ctx.get_identity_name():
            return id

    return -1


def get_num_partitions(fl_ctx: FLContext):
    """Get the number of partitions based on the number of clients."""
    engine = fl_ctx.get_engine()
    return len(engine.get_clients())


def _validate_flower_executable(executable_name: str, executable_path: str):
    """Validate that a Flower executable exists and provide helpful error message if not.

    Args:
        executable_name: Name of the executable (e.g., FLOWER_SUPERLINK)
        executable_path: Full path to the executable

    Raises:
        RuntimeError: If the executable is not found with installation instructions
    """
    if not os.path.isfile(executable_path):
        error_msg = (
            f"Flower executable '{executable_name}' not found at: {executable_path}\n"
            f"\n"
            f"This indicates Flower is not properly installed in your Python environment.\n"
            f"Please install a compatible Flower version:\n"
            f"  pip install 'flwr>=1.43'\n"
            f"\n"
            f"If using a virtual environment, ensure it's activated before installation.\n"
            f"Current Python: {sys.executable}"
        )
        raise RuntimeError(error_msg)

    # Check if executable has execute permissions
    if not os.access(executable_path, os.X_OK):
        error_msg = (
            f"Flower executable '{executable_name}' found but not executable: {executable_path}\n"
            f"Please ensure the file has execute permissions:\n"
            f"  chmod +x {executable_path}"
        )
        raise RuntimeError(error_msg)


def _validate_flwr_runtime():
    # Check on every deployed site, not only where a Recipe is authored.
    from nvflare.app_opt.flower.recipe import _validate_flwr_version

    _validate_flwr_version()


def _wait_for_superlink(address, process, timeout):
    host, port = split_address(address)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Flower SuperLink exited during startup; see superlink_log.txt")
        conn = http.client.HTTPConnection(host, port, timeout=min(1.0, max(0.01, deadline - time.monotonic())))
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            if response.status == 200:
                return
        except (OSError, http.client.HTTPException):
            pass
        finally:
            conn.close()
        time.sleep(0.1)
    raise RuntimeError(f"Flower SuperLink was not healthy within {timeout} seconds")


def _format_run_config_value(value) -> str:
    """Format a Flower run_config value as a TOML-compatible scalar literal."""
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, str):
        return json.dumps(value)

    raise TypeError(f"invalid run_config value type {type(value)}: values must be bool, int, float, or str")


class FlowerClientApplet(CLIApplet):
    def __init__(
        self,
        extra_env: dict = None,
        allow_runtime_dependency_installation: bool = False,
    ):
        """Constructor of FlowerClientApplet, which extends CLIApplet.

        Note: flower_app_path is not used on clients - the Flower app is distributed
        from the server via Flower's FAB mechanism.
        """
        CLIApplet.__init__(self, stop_method="term")
        self.allow_runtime_dependency_installation = allow_runtime_dependency_installation

        # Ensure PATH includes the venv bin directory so Flower's internal
        # subprocesses (flower-superexec, etc.) can find executables
        python_bin_dir = os.path.dirname(sys.executable)
        if extra_env is None:
            extra_env = {}

        # Add venv bin directory to PATH
        current_path = os.environ.get("PATH", "")
        if python_bin_dir not in current_path:
            extra_env["PATH"] = f"{python_bin_dir}{os.pathsep}{current_path}"

        self.extra_env = extra_env.copy()

    def start(self, app_ctx: dict):
        _validate_flwr_runtime()
        try:
            self._proc_mgr = start_process(self.get_command(app_ctx), app_ctx[Constant.APP_CTX_FL_CONTEXT])
        except Exception:
            self._start_error = True
            raise

    def stop(self, timeout=5.0):
        manager = self._proc_mgr
        self._proc_mgr = None
        return manager.stop(timeout) if manager else 0

    def get_command(self, ctx: dict) -> CommandDescriptor:
        """Implementation of the get_command method required by the super class CLIApplet.
        It returns the CLI command for starting Flower's client app, as well as the full path of the log file
        for the client app.

        Args:
            ctx: the applet run context

        Returns: CLI command for starting client app and name of log file.

        """
        superlink_addr = ctx.get(Constant.APP_CTX_SUPERLINK_ADDR)
        clientapp_api_addr = ctx.get(Constant.APP_CTX_CLIENTAPP_API_ADDR)
        fl_ctx = ctx.get(Constant.APP_CTX_FL_CONTEXT)
        if not isinstance(fl_ctx, FLContext):
            self.logger.error(f"expect APP_CTX_FL_CONTEXT to be FLContext but got {type(fl_ctx)}")
            raise RuntimeError("invalid FLContext")

        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        if not isinstance(ws, Workspace):
            self.logger.error(f"expect workspace to be Workspace but got {type(ws)}")
            raise RuntimeError("invalid workspace")

        job_id = fl_ctx.get_job_id()
        app_dir = ws.get_app_dir(job_id)

        # Get the full path to flower-supernode from the current Python environment
        python_bin_dir = os.path.dirname(sys.executable)
        flower_supernode_path = os.path.join(python_bin_dir, FLOWER_SUPERNODE)

        # Validate that flower-supernode is installed and executable
        _validate_flower_executable(FLOWER_SUPERNODE, flower_supernode_path)

        host, port = split_address(clientapp_api_addr)
        split_address(superlink_addr)
        cmd = shlex.join(
            [
                flower_supernode_path,
                "--insecure",
                "--superlink",
                superlink_addr,
                "--host",
                host,
                "--port",
                str(port),
            ]
        )
        if self.allow_runtime_dependency_installation:
            cmd += " --allow-runtime-dependency-installation"

        # add node config
        node_config_str = self._get_node_config(fl_ctx)
        if node_config_str:
            cmd += node_config_str

        env = self.extra_env.copy()
        env["FLWR_HOME"] = os.path.join(ws.get_run_dir(job_id), "flwr_home")
        env[ANALYTICS_BOOTSTRAP_ENV] = os.path.join(ws.get_app_config_dir(job_id), ANALYTICS_BOOTSTRAP_FILE)

        # Use app_dir as the cwd for Flower's client app and its project files.
        self.logger.info(f"starting flower client app: {cmd}")
        return CommandDescriptor(
            cmd=cmd,
            cwd=app_dir,
            env=env,
            log_file_name="client_app_log.txt",
            stdout_msg_prefix="FLWR-CA",
            stop_method=StopMethod.TERMINATE,
        )

    def _get_node_config(self, fl_ctx: FLContext):
        """Get the node config for the flower client app."""
        try:
            cmd = f"client-name={json.dumps(fl_ctx.get_identity_name())}"
            partition_id = get_partition_id(fl_ctx)
            if partition_id != -1:
                cmd += f" partition-id={partition_id}"
            cmd += f" num-partitions={get_num_partitions(fl_ctx)}"
            return " --node-config " + shlex.quote(cmd)
        except Exception as ex:
            self.log_error(fl_ctx, f"Exception getting node configuration from fl_ctx: {secure_format_exception(ex)}")
            return None


class FlowerServerApplet(Applet):
    def __init__(
        self,
        database: str,
        superlink_ready_timeout: float,
        superlink_grace_period=1.0,
        superlink_min_query_interval=10.0,
        run_config: Optional[dict] = None,
        allow_runtime_dependency_installation: bool = False,
        flower_app_path: Optional[str] = None,
    ):
        """Constructor of FlowerServerApplet.

        Args:
            database: database spec to be used by the server app
            superlink_ready_timeout: how long to wait for the superlink process to become ready
            superlink_grace_period: how long to wait for superlink to gracefully shutdown
            superlink_min_query_interval: minimal interval for querying superlink for status
            run_config: optional dict for flwr run --run-config arguments
            allow_runtime_dependency_installation: whether to allow dynamic dependency installation
            flower_app_path: absolute path to pre-deployed Flower app on the server (clients receive via FAB)
        """
        Applet.__init__(self)
        self._superlink_process_mgr = None
        self.database = database
        self.run_config = run_config
        self.superlink_ready_timeout = superlink_ready_timeout
        self.superlink_grace_period = superlink_grace_period
        self.superlink_min_query_interval = superlink_min_query_interval
        self.allow_runtime_dependency_installation = allow_runtime_dependency_installation
        self.flower_app_path = flower_app_path
        self.run_id = None
        self.last_query_time = None
        self.last_check_status = None
        self.last_check_stopped = False
        self.flower_app_dir = None
        self.exec_api_addr = None
        self.flwr_home_dir = None
        self.run_dir = None
        self.flower_run_finished = False
        self.flwr_stop_called = False  # have we called 'flwr stop'?
        self.flower_run_rc = None

        self._start_error = False
        self.stop_lock = threading.Lock()

    def _start_process(self, name: str, cmd_desc: CommandDescriptor, fl_ctx: FLContext) -> ProcessManager:
        self.logger.info(f"starting {name}: {cmd_desc.cmd}")
        try:
            return start_process(cmd_desc, fl_ctx, stop_method="term")
        except Exception as ex:
            self.logger.error(f"exception starting applet: {secure_format_exception(ex)}")
            self._start_error = True

    def start(self, app_ctx: dict):
        """Start the applet.

        Flower requires two processes for server application:
            superlink: this process is responsible for client communication
            server_app: this process performs server side of training.

        We start the superlink first, and wait for it to become ready, then start the server app.
        Each process will have its own log file in the job's run dir. The superlink's log file is named
        "superlink_log.txt". The server app's log file is named "server_app_log.txt".

        Args:
            app_ctx: the run context of the applet.

        Returns:

        """
        # try to start superlink first
        _validate_flwr_runtime()
        superlink_addr = app_ctx[Constant.APP_CTX_SUPERLINK_ADDR]
        fl_ctx = app_ctx.get(Constant.APP_CTX_FL_CONTEXT)
        if not isinstance(fl_ctx, FLContext):
            self.logger.error(f"expect APP_CTX_FL_CONTEXT to be FLContext but got {type(fl_ctx)}")
            raise RuntimeError("invalid FLContext")

        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        if not isinstance(ws, Workspace):
            self.logger.error(f"expect workspace to be Workspace but got {type(ws)}")
            raise RuntimeError("invalid workspace")

        if self.flower_app_path:
            # Resolve relative path to absolute path relative to workspace root
            workspace_root = ws.get_root_dir()
            self.flower_app_dir = os.path.abspath(os.path.join(workspace_root, self.flower_app_path))

            # Validate path format
            validate_flower_app_path(self.flower_app_path)

            # Check for symlinks on the resolved absolute path
            validate_flower_app_path_no_symlinks(self.flower_app_dir)

            # Check filesystem existence
            if not os.path.isdir(self.flower_app_dir):
                raise RuntimeError(
                    f"flower_app_path '{self.flower_app_path}' does not exist on this host. "
                    "Ensure the Flower app is pre-deployed on the server. "
                    "(Clients receive the app from the server via Flower's FAB distribution)."
                )
        else:
            custom_dir = ws.get_app_custom_dir(fl_ctx.get_job_id())
            self.flower_app_dir = custom_dir
        self.exec_api_addr = superlink_addr
        self.run_dir = ws.get_run_dir(fl_ctx.get_job_id())
        self.flwr_home_dir = self._prepare_flwr_home(ws.get_run_dir(fl_ctx.get_job_id()))

        db_arg = ""
        if self.database:
            db_path = self.database
            if not os.path.isabs(db_path) and db_path not in (":memory:", ":flwr-in-memory:"):
                db_path = os.path.abspath(db_path)
            db_arg = f"--database {shlex.quote(db_path)}"

        # Get the full path to flower-superlink from the current Python environment
        python_bin_dir = os.path.dirname(sys.executable)
        flower_superlink_path = os.path.join(python_bin_dir, FLOWER_SUPERLINK)

        # Validate that flower-superlink is installed and executable
        _validate_flower_executable(FLOWER_SUPERLINK, flower_superlink_path)

        # Ensure PATH includes venv bin directory for Flower's internal subprocesses
        # (flower-superlink internally spawns flower-superexec which needs to be in PATH)
        env = self._build_flower_env(include_flwr_home=True)

        host, port = split_address(superlink_addr)
        superlink_cmd = f"{shlex.quote(flower_superlink_path)} --insecure {db_arg} " f"--host {host} --port {port}"
        superlink_cmd += (
            " --allow-runtime-dependency-installation"
            if self.allow_runtime_dependency_installation
            else " --disable-runtime-dependency-installation"
        )

        run_dir = ws.get_run_dir(fl_ctx.get_job_id())
        if not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=True)

        cmd_desc = CommandDescriptor(
            cmd=superlink_cmd,
            cwd=run_dir,
            env=env,
            log_file_name="superlink_log.txt",
            stdout_msg_prefix="FLWR-SL",
            stop_method=StopMethod.TERMINATE,
        )

        self._superlink_process_mgr = self._start_process(name="superlink", cmd_desc=cmd_desc, fl_ctx=fl_ctx)
        if not self._superlink_process_mgr:
            raise RuntimeError("cannot start superlink process")

        _wait_for_superlink(superlink_addr, self._superlink_process_mgr, self.superlink_ready_timeout)

        # submitting the server app using "flwr run" command
        flwr_run_cmd = self._flower_command("run")
        run_info = self._run_flower_command(flwr_run_cmd, cwd=self.flower_app_dir)
        run_id = run_info.get("run-id")
        if not run_id:
            raise RuntimeError(f"invalid result from command '{flwr_run_cmd}': missing run-id")

        self.logger.info(f"submitted Flower App and got run id {run_id}")
        self.run_id = run_id

    def _build_flower_env(self, include_flwr_home: bool) -> Optional[dict]:
        python_bin_dir = os.path.dirname(sys.executable)
        current_path = os.environ.get("PATH", "")
        env = {}
        if python_bin_dir not in current_path:
            env["PATH"] = f"{python_bin_dir}{os.pathsep}{current_path}"

        if include_flwr_home and self.flwr_home_dir:
            env["FLWR_HOME"] = self.flwr_home_dir

        return env if env else None

    def _build_flower_config(self) -> str:
        if not self.exec_api_addr:
            raise RuntimeError("Flower control API address is not set")

        return (
            "[superlink]\n"
            f'default = "{FLOWER_SUPERLINK_CONNECTION}"\n'
            "\n"
            f"[superlink.{FLOWER_SUPERLINK_CONNECTION}]\n"
            f'address = "{self.exec_api_addr}"\n'
            # The Flower control channel stays local to the NVFlare server host,
            # so the generated connection intentionally matches the colocated
            # SuperLink's non-TLS startup mode.
            "insecure = true\n"
        )

    def _prepare_flwr_home(self, run_dir: str) -> str:
        flwr_home_dir = os.path.join(run_dir, "flwr_home")
        os.makedirs(flwr_home_dir, exist_ok=True)
        config_path = os.path.join(flwr_home_dir, FLOWER_CONFIG_FILE)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(self._build_flower_config())
        self.logger.info(f"wrote Flower configuration to {config_path}")
        return flwr_home_dir

    def _flower_command(self, cmd_name: str, cmd_args=""):
        # Get the full path to flwr from the current Python environment
        python_bin_dir = os.path.dirname(sys.executable)
        flwr_path = os.path.join(python_bin_dir, FLOWER_CLI)

        # Validate that flwr is installed and executable
        _validate_flower_executable(FLOWER_CLI, flwr_path)

        normalized_cmd_name = "list" if cmd_name == "ls" else cmd_name
        command_parts = [shlex.quote(flwr_path), normalized_cmd_name]
        if normalized_cmd_name == "run":
            command_parts.extend([shlex.quote("."), shlex.quote(FLOWER_SUPERLINK_CONNECTION)])
        elif normalized_cmd_name == "stop":
            if not cmd_args:
                raise ValueError("stop command requires a run ID")
            command_parts.extend([shlex.quote(cmd_args), shlex.quote(FLOWER_SUPERLINK_CONNECTION)])
        elif normalized_cmd_name == "list":
            if cmd_args:
                command_parts.extend(["--run-id", shlex.quote(str(cmd_args))])
            command_parts.append(shlex.quote(FLOWER_SUPERLINK_CONNECTION))
        elif cmd_args:
            command_parts.append(shlex.quote(cmd_args))

        if self.run_config and cmd_name == "run":
            for key, value in self.run_config.items():
                serialized = f"{key}={_format_run_config_value(value)}"
                command_parts.extend(["--run-config", shlex.quote(serialized)])

        command_parts.extend(["--format", "json"])

        return " ".join(command_parts)

    def _run_flower_command(self, command: str, cwd: Optional[str] = None):
        self.logger.debug(f"running flower command: {command}")
        cmd_desc = CommandDescriptor(cmd=command, env=self._build_flower_env(include_flwr_home=True), cwd=cwd)
        reply = run_command(cmd_desc)
        if not isinstance(reply, str):
            raise RuntimeError(f"failed to run command '{command}': expect reply to be str but got {type(reply)}")

        self.logger.debug(f"flower command {command}: {reply=}")
        # the reply must be a json str
        try:
            result = json.loads(reply)
        except Exception as ex:
            err = f"invalid result from command '{command}': {secure_format_exception(ex)}"
            self.logger.error(err)
            raise RuntimeError(err)

        if not isinstance(result, dict):
            err = f"invalid result from command '{command}': expect dict but got {type(result)}"
            self.logger.error(err)
            raise RuntimeError(err)

        success = result.get("success", False)
        if not success:
            err = f"failed command '{command}': {success=} {result=}"
            self.logger.error(err)
            raise RuntimeError(err)

        self.logger.debug(f"result of {command}: {result}")
        return result

    @staticmethod
    def _stop_process(p: ProcessManager) -> int:
        if not p:
            # nothing to stop
            return 0
        else:
            return p.stop()

    def stop(self, timeout=0.0) -> int:
        """Stop the server applet's superlink.

        Args:
            timeout: how long to wait before forcefully stopping (kill) the process.

        Note: we always stop the process immediately - do not wait for the process to stop itself.

        Returns:

        """
        with self.stop_lock:
            if self.run_id and not self.flwr_stop_called and not self.flower_run_finished:
                # stop the server app
                # we may not be able to issue 'flwr stop' more than once!
                self.flwr_stop_called = True
                flwr_stop_cmd = self._flower_command("stop", self.run_id)
                self.logger.info(f"Issued command to stop Flower App: {flwr_stop_cmd}")
                try:
                    self._run_flower_command(flwr_stop_cmd)
                except Exception as ex:
                    # ignore exception
                    self.logger.error(f"exception running '{flwr_stop_cmd}': {secure_format_exception(ex)}")

                # wait a while to let superlink and supernodes gracefully stop the app
                time.sleep(self.superlink_grace_period)

            # Flower 1.43 stores ServerApp logs in SuperLink. Preserve them
            # before shutting down its default in-memory database.
            if self.run_id and self._superlink_process_mgr and self._superlink_process_mgr.poll() is None:
                self._save_run_log()

            # stop the superlink
            self._stop_process(self._superlink_process_mgr)
            self._superlink_process_mgr = None
            return 0

    def _save_run_log(self):
        try:
            executable = os.path.join(os.path.dirname(sys.executable), FLOWER_CLI)
            command = shlex.join([executable, "log", str(self.run_id), FLOWER_SUPERLINK_CONNECTION, "--show"])
            output = run_command(CommandDescriptor(cmd=command, env=self._build_flower_env(include_flwr_home=True)))
            with open(os.path.join(self.run_dir, "server_app_log.txt"), "w", encoding="utf-8") as log:
                log.write(output)
        except Exception as ex:
            self.logger.warning(f"could not save Flower ServerApp log: {secure_format_exception(ex)}")

    @staticmethod
    def _is_process_stopped(p: ProcessManager):
        if p:
            return_code = p.poll()
            if return_code is None:
                return False, 0
            else:
                return True, return_code
        else:
            return True, 0

    def _check_flower_run_status(self):
        if not self.last_query_time or time.time() - self.last_query_time > self.superlink_min_query_interval:
            # time to query
            self.last_query_time = time.time()
            self.last_check_stopped, self.last_check_status = self._query_for_run_status()

        return self.last_check_stopped, self.last_check_status

    def _query_for_run_status(self):
        # check whether the app is finished
        flwr_ls_cmd = self._flower_command("list", str(self.run_id))
        try:
            run_info = self._run_flower_command(flwr_ls_cmd)
        except Exception as ex:
            self.logger.error(f"exception running '{flwr_ls_cmd}': {secure_format_exception(ex)}")
            return True, TieConstant.EXIT_CODE_FATAL_ERROR

        runs = run_info.get("runs")
        if not runs:
            # the app is no longer there
            self.logger.error(f"invalid result from command '{flwr_ls_cmd}': missing run info")
            return True, TieConstant.EXIT_CODE_FATAL_ERROR

        if not isinstance(runs, list):
            self.logger.error(f"invalid result from command '{flwr_ls_cmd}': expect run list but got {type(runs)}")
            return True, TieConstant.EXIT_CODE_FATAL_ERROR

        run = runs[0]
        if not isinstance(run, dict):
            self.logger.error(f"invalid result from command '{flwr_ls_cmd}': expect run to be dict but got {type(run)}")
            return True, TieConstant.EXIT_CODE_FATAL_ERROR

        status = run.get("status")
        if not status:
            self.logger.error(f"invalid result from command '{flwr_ls_cmd}': missing status from {run}")
            return True, TieConstant.EXIT_CODE_FATAL_ERROR

        if not isinstance(status, str):
            self.logger.error(f"invalid result from command '{flwr_ls_cmd}': bad status value '{status}'")
            return True, TieConstant.EXIT_CODE_FATAL_ERROR

        if status.startswith("finished"):
            self.logger.info(f"Flower Run {self.run_id} finished: {status=}")
            self.flower_run_finished = True
            if status.endswith("completed"):
                rc = 0
            else:
                rc = TieConstant.EXIT_CODE_FAILED
            self.flower_run_rc = rc
            return True, rc
        else:
            return False, 0

    def is_stopped(self) -> (bool, int):
        """Check whether the server applet is already stopped

        Returns: a tuple of: whether the applet is stopped, exit code if stopped.

        Note: if either superlink or server app is stopped, we treat the applet as stopped.

        """
        if self._start_error:
            return True, TieConstant.EXIT_CODE_CANT_START

        superlink_stopped, superlink_rc = self._is_process_stopped(self._superlink_process_mgr)
        if superlink_stopped:
            return True, superlink_rc

        if self.flower_run_finished:
            return True, self.flower_run_rc

        if self.last_check_stopped:
            return self.last_check_stopped, self.last_check_status

        with self.stop_lock:
            self.last_check_stopped, self.last_check_status = self._check_flower_run_status()

        return self.last_check_stopped, self.last_check_status
