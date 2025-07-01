from typing import Optional, List, Dict, Any, Union
from nvflare.apis.fl_api.strategy.strategy import Strategy
from nvflare.apis.fl_api.trainer.fed_trainer import FedTrainer
from nvflare.apis.fl_api.env.deployment_envs import DeploymentEnv, SimulationEnv


class FLJob:
    def __init__(
            self,
            strategy: Strategy,
            client_trainers: Dict[str, FedTrainer],
            rounds: int = 1,
            deployment_settings: Optional[Dict[str, Any]] = None,
    ):
        self.strategy = strategy
        self.client_trainers = client_trainers
        self.rounds = rounds
        self.deployment_settings = deployment_settings or {}
        self.connected_clients = list(client_trainers.keys())

    def fit(self, env: Optional[DeploymentEnv] = None):
        if env is None:
            env = SimulationEnv()
        return env.run()
