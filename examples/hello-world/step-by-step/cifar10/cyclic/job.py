
import sys

code_path = "../code/fl"
if code_path not in sys.path:
    sys.path.append(code_path)

from net import Net
from nvflare import FedJob
from nvflare.app_common.workflows.cyclic import Cyclic
from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner


if __name__ == "__main__":
    n_clients = 2
    num_rounds = 3
    train_script = "../code/fl/train.py"

    job = FedJob(name="cyclic")

    # Define the controller workflow and send to server
    controller = Cyclic(
        num_clients=n_clients,
        num_rounds=num_rounds,
    )
    job.to(controller, "server")

    # Define the initial global model and send to server
    job.to(PTModel(Net()), "server")

    # Add clients
    for i in range(n_clients):
        executor = ScriptRunner(
            script=train_script,
            script_args="",
        )
        job.to(executor, f"site-{i+1}")

    job.export_job("/tmp/nvflare/jobs")
    job.simulator_run("/tmp/nvflare/jobs/workdir", gpu="0")
