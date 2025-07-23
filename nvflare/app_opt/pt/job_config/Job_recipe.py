from abc import ABC
from typing import Optional, Callable

from build.lib.nvflare.job_config.script_runner import FrameworkType
from nvflare import FedJob
from nvflare.app_common.workflows.cyclic import Cyclic


class ExecEnv(ABC):
    def setup(self):
        pass

class SimEnv(ExecEnv):
    def __init__(self, workspace_dir: str = "/tmp/nvflare/sim/workspace"):
        self.workspace_dir = workspace_dir


class JobRecipe(ABC):

    def get_job(self) -> FedJob:
        raise NotImplemented

    def execute(self, env):

        if env is None:
            env = SimEnv()

        if isinstance(env, SimEnv):
            job: FedJob = self.get_job()
            job.export_job("/tmp/nvflare/jobs/job_config")
            job.simulator_run(workspace=env.workspace_dir, gpu="0")


        pass



from nvflare.app_opt.pt.job_config.fed_avg import FedAvgJob
from nvflare.job_config.script_runner import ScriptRunner


class FedAvgRecipe(JobRecipe):
    def __init__(self,
                 clients:int,  # [“site-1”, “site-2”, ..]
                 num_rounds:int,
                 client_script: str,
                 model = None,
                 client_script_args="--local_epochs 1",
                 aggregate_fn: Optional[Callable] = None,
                 ):
        self.n_clients = clients
        self.num_rounds = num_rounds
        self.client_script = client_script
        self.model = model
        self.client_script_args = client_script_args
        self.aggregate_fn = aggregate_fn
        self.recipe_name = "fed_avg"

    #     todo:
    #     if model is None:
    #         model = self.load_model()


    def get_job(self) -> FedJob:

        job = FedAvgJob(
            name=self.recipe_name,
            n_clients= self.n_clients,
            num_rounds=self.num_rounds,
            initial_model = self.model,
        )
        # Add clients
        for i in range(self.n_clients):
            executor = ScriptRunner(
                script = self.client_script,
                script_args=self.client_script_args
            )
            job.to(executor, f"site-{i + 1}")

        return job


class CyclicRecipe(JobRecipe):
    def __init__(self,
                 clients:int,  # [“site-1”, “site-2”, ..]
                 num_rounds:int,
                 client_script: str,
                 model = None,
                 client_script_args="--local_epochs 1",
                 aggregate_fn: Optional[Callable] = None,
                 framework : Optional[FrameworkType] = FrameworkType.TENSORFLOW,
                 ):
        self.n_clients = clients
        self.num_rounds = num_rounds
        self.client_script = client_script
        self.model = model
        self.client_script_args = client_script_args
        self.aggregate_fn = aggregate_fn
        self.recipe_name = "cyclic"
        self.framework = framework


    def get_job(self) -> FedJob:

        job = FedJob(name="cyclic")
        # Define the controller workflow and send to server
        controller = Cyclic(
            num_clients=self.n_clients,
            num_rounds=self.num_rounds,
        )
        job.to(controller, "server")

        # Define the initial global model and send to server
        job.to(self.model, "server")

        # Add clients
        for i in range(self.n_clients):
            executor = ScriptRunner(
                script=self.client_script,
                script_args=self.client_script_args,
                framework=self.framework,
            )
            job.to(executor, f"site-{i + 1}")
        return job


