from typing import Any, Dict, Optional
import torch
from torch import nn
from nvflare.apis.fl_api.trainers.base.fed_trainer import FedTrainer


class PyTorchTrainer(FedTrainer):
    """FedTrainer for PyTorch models."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[torch.nn.Module] = None,
        train_loader: Optional[Any] = None,
        val_loader: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model.to(self.device)
        
        super().__init__(
            local_trainer=self,
            get_state_fn=self._get_model_state,
            set_state_fn=self._set_model_state,
            **kwargs
        )
    
    def _get_model_state(self) -> Dict[str, Any]:
        return {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None
        }
    
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        self.model.load_state_dict(state['model_state'])
        if self.optimizer and state['optimizer_state']:
            self.optimizer.load_state_dict(state['optimizer_state'])
    
    def fit(self):
        self.model.train()
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
        return loss.item()

