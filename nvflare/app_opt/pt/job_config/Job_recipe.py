from abc import ABC
from typing import Optional, Callable

class ExecEnv(ABC):
    def setup(self):
        pass

class SimEnv(ExecEnv):
    def __init__(self, workspace_dir: str = "/tmp/nvflare/sim/workspace"):
        self.workspace_dir = workspace_dir


class JobRecipe(ABC):
    def execute(self, env):
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
        self.recipe_name = "hello-pt"

    #     todo:
    #     if model is None:
    #         model = self.load_model()

    def execute(self, env: Optional[ExecEnv] = None):

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

        if env is None:
            env = SimEnv()

        if isinstance(env, SimEnv):
            job.export_job("/tmp/nvflare/jobs/job_config")
            job.simulator_run(workspace=env.workspace_dir, gpu="0")

