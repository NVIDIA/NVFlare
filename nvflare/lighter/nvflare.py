import argparse
import sys
import os
import random
from nvflare.lighter.poc import prepare_poc as generate_poc


def prepare_poc(n_clients: int, poc_workspace: str):
    print(f"prepare_poc at {poc_workspace} for {n_clients} clients")
    generate_poc(n_clients, poc_workspace)


def sort_service_cmds(service_cmds: list) -> list:
    def sort_first(val):
        return val[0]
    order_services = []
    for service_name, cmd_path in service_cmds:
        if service_name == "server":
            order_services.append((0, service_name, cmd_path))
        elif service_name == "admin":
            order_services.append((sys.maxsize, service_name, cmd_path))
        else:
            order_services.append((random.randint(2, len(service_cmds)), service_name, cmd_path))

    order_services.sort(key=sort_first)
    return [(service_name, cmd_path) for n, service_name, cmd_path in order_services]


def get_cmd_path(poc_workspace, service_name, cmd):
    # todo: add variable arguments if needed
    service_dir = os.path.join(poc_workspace, service_name)
    bin_dir = os.path.join(service_dir, "startup")
    cmd_path = os.path.join(bin_dir, cmd)
    return cmd_path


def get_service_cmd(cmd_type: str, service_name: str) -> str:
    cmd = None
    if cmd_type == "start":
        if service_name == "admin":
            cmd = "fl_admin.sh"
        else:
            cmd = "start.sh"
    elif cmd_type == "stop":
        # todo why there is no stop admin cmd
        if service_name != "admin":
            cmd = "stop_fl.sh"
    else:
        print(f"Invalid command type for service {service_name}, expecting start or stop")
        sys.exit(5)

    return cmd


def is_poc_ready(poc_workspace: str):
    # check server and admin directories exist
    admin_dir = os.path.join(poc_workspace, "admin")
    server_dir = os.path.join(poc_workspace, "server")
    overseer_dir = os.path.join(poc_workspace, "overseer")
    return os.path.isdir(server_dir) and os.path.isdir(admin_dir) and os.path.isdir(overseer_dir)


def start_poc(poc_workspace: str):
    print(f"start_poc at {poc_workspace}")
    if not is_poc_ready(poc_workspace):
        print(f"workspace {poc_workspace} is not ready, please use poc --prepare to prepare poc workspace")
        sys.exit(2)
    _run_poc("start", poc_workspace, excluded=['overseer'])


def stop_poc(poc_workspace: str):
    print(f"stop_poc at {poc_workspace}")
    if not is_poc_ready(poc_workspace):
        print(f"invalid workspace {poc_workspace}")
        sys.exit(4)
    _run_poc("stop", poc_workspace, excluded=['overseer'])


def _build_commands(cmd_type: str, poc_workspace: str, excluded: list):
    service_commands = []
    for root, dirs, files in os.walk(poc_workspace):
        if root == poc_workspace:
            for d in dirs:
                if d not in excluded:
                    cmd = get_service_cmd(cmd_type, d)
                    if cmd:
                        service_commands.append((d, get_cmd_path(root, d, cmd)))
    return sort_service_cmds(service_commands)


def _run_poc(cmd_type: str, poc_workspace: str, excluded: list):
    service_commands = _build_commands(cmd_type, poc_workspace, excluded)
    import time
    for service_name, cmd_path in service_commands:
        print(f"{cmd_type}: service: {service_name}, executing {cmd_path}")
        import subprocess
        if service_name == 'admin':
            subprocess.run([cmd_path])
        else:
            subprocess.Popen([cmd_path])
            time.sleep(3)


def clean_poc(poc_workspace: str):
    import shutil
    if is_poc_ready(poc_workspace):
        shutil.rmtree(poc_workspace, ignore_errors=True)
    else:
        print(f"{poc_workspace} is not valid poc directory")
        exit(1)


def def_poc_parser(sub_cmd, prog_name: str):
    poc_parser = sub_cmd.add_parser('poc')
    poc_parser.add_argument('-n', '--n_clients', type=int, nargs='?', default=2, help='number of sites or clients')
    poc_parser.add_argument('-d', '--workspace', type=str, nargs='?', default=f"/tmp/{prog_name}/poc",
                            help='poc workspace directory')
    poc_parser.add_argument('--prepare', dest='prepare_poc', action='store_const', const=prepare_poc,
                            help='prepare poc workspace')
    poc_parser.add_argument('--start', dest='start_poc', action='store_const', const=start_poc, help='start poc')
    poc_parser.add_argument('--stop', dest='stop_poc', action='store_const', const=stop_poc, help='stop poc')
    poc_parser.add_argument('--clean', dest='clean_poc', action='store_const', const=clean_poc, help='cleanup poc workspace')


def is_poc(cmd_args) -> bool:
    return hasattr(cmd_args, 'start_poc') or \
           hasattr(cmd_args, 'prepare_poc') or \
           hasattr(cmd_args, 'stop_poc') or \
           hasattr(cmd_args, 'clean_poc')


def is_provision(cmd_args) -> bool:
    #  todo add provision handling
    return False


def handle_poc_cmd(cmd_args):
    if cmd_args.start_poc:
        cmd_args.start_poc(cmd_args.workspace)
    elif cmd_args.prepare_poc:
        cmd_args.prepare_poc(cmd_args.n_clients, cmd_args.workspace)
    elif cmd_args.stop_poc:
        cmd_args.stop_poc(cmd_args.workspace)
    elif cmd_args.clean_poc:
        cmd_args.clean_poc(cmd_args.workspace)
    else:
        print(f"unable to handle poc command:{cmd_args}")
        sys.exit(3)


def handle_provision_cmd(cmd_args):
    print("handle provision command")
    pass


def def_provision_parser(sub_cmd, prog_name: str):
    provision_parser = sub_cmd.add_parser('provision')
    provision_parser.add_argument('-n', '--n_clients', type=int, nargs='?', default=2, help='number of sites or clients')


def parse_args(prog_name: str):
    _parser = argparse.ArgumentParser(description='nvflare parser')
    sub_cmd = _parser.add_subparsers(description="sub command parser")
    def_poc_parser(sub_cmd, prog_name)
    def_provision_parser(sub_cmd, prog_name)
    return _parser, _parser.parse_args()


def run(job_name):
    cwd = os.getcwd()
    sys.path.append(cwd)
    prog_parser, prog_args = parse_args(job_name)

    if is_poc(prog_args):
        handle_poc_cmd(prog_args)
    elif is_provision(prog_args):
        handle_provision_cmd(prog_args)
    else:
        prog_parser.print_help()


def main():
    run("nvflare")


if __name__ == "__main__":
    main()
