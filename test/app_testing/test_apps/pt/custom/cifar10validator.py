import torch
import torchvision.datasets
from torchvision import transforms

from net import SimpleNetwork
from nvflare.apis.dxo import from_shareable, DataKind, DXO
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants


class Cifar10Validator(Executor):
    
    def __init__(self, validate_task_name=AppConstants.TASK_VALIDATION):
        super(Cifar10Validator, self).__init__()

        self._validate_task_name = validate_task_name

        # Setup the model
        self.model = SimpleNetwork()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Preparing the dataset for testing.
        self.test_data = torchvision.datasets.CIFAR10(root='~/data', train=False, transform=self.transforms)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=4, shuffle=False)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        out_shareable = Shareable()
        if task_name == self._validate_task_name:
            try:
                dxo = from_shareable(shareable)

                # Check if dxo is valid.
                if not dxo:
                    self.log_exception(fl_ctx, "DXO invalid")
                    out_shareable.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
                    return out_shareable

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                    out_shareable.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
                    return out_shareable

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                weights = dxo.data
                weights = {k: torch.as_tensor(v, device=self.device) for k, v in weights.items()}

                # Get validation accuracy
                val_accuracy = self.do_validation(weights)
                self.log_info(fl_ctx, f"Accuracy when validating {model_owner}'s model on"
                                      f" {fl_ctx.get_identity_name()}"f's data: {val_accuracy}')

                dxo = DXO(data_kind=DataKind.METRICS, data={'val_acc': val_accuracy})
                return dxo.to_shareable()
            except:
                self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                out_shareable.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
                return out_shareable
        else:
            out_shareable.set_return_code(ReturnCode.TASK_UNKNOWN)
            return out_shareable

    def do_validation(self, weights):
        self.model.load_state_dict(weights)

        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)

                _, pred_label = torch.max(output, 1)

                correct += (pred_label == labels).sum().item()
                total += images.size()[0]

            metric = correct/float(total)

        return metric
