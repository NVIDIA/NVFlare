import torch
from nvflare.apis.dxo import from_shareable

from .gradient_tracking import GTExecutor


class GTADAMExecutor(GTExecutor):
    def _pre_algorithm_run(self, fl_ctx, shareable, abort_signal):
        super()._pre_algorithm_run(fl_ctx, shareable, abort_signal)

        data = from_shareable(shareable).data
        self.beta1 = data["beta1"]
        self.beta2 = data["beta2"]
        self.epsilon = data["epsilon"]
        self.G = torch.tensor(1e6)
        self.m = [torch.zeros_like(param) for param in self.model.parameters()]
        self.v = [torch.zeros_like(param) for param in self.model.parameters()]


    def _update_local_state(self, stepsize):
        for i in range(len(self.tracker)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * self.tracker[i]
            self.v[i] = torch.minimum(
                self.beta2 * self.v[i] + (1 - self.beta2) * self.tracker[i] ** 2, self.G
            )
        
        with torch.no_grad():
            for idx, param in enumerate(self.model.parameters()):
                if param.requires_grad:
                    descent = torch.divide(
                        self.m[idx], torch.sqrt(self.v[idx] + self.epsilon)
                    )
                    param.add_(descent, alpha=-stepsize)
