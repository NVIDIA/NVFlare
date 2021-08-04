import torch
import torch.nn as nn
import torch.optim as optim
from net import Net
from nvflare.apis import Trainer
from nvflare.apis.fl_constant import FLConstants, ShareableKey, ShareableValue
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from torchvision import datasets, transforms


class SimpleTrainer(Trainer):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Net()
        self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.trainset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.transform
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.trainset, batch_size=4, shuffle=True, num_workers=2
        )

    def local_train(self):
        """A regular training procedure.

        :return:
        """
        self.model.train()
        for epoch in range(2):
            # set the model to train mode
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def train(self, shareable: Shareable, fl_ctx: FLContext, abort_signal) -> Shareable:
        """
        This function is an extended function from the super class.
        As a supervised learning based trainer, the train function will run
        evaluate and train engines based on model weights from `shareable`.
        After finishing training, a new `Shareable` object will be submitted
        to server for aggregation.

        Args:
            shareable: the `Shareable` object acheived from server.
            fl_ctx: the `FLContext` object achieved from server.
            abort_signal: if triggered, the training will be aborted.

        Returns:
            a new `Shareable` object to be submitted to server for aggregation.
        """
        # retrieve model weights download from server's shareable
        model_weights = shareable[ShareableKey.MODEL_WEIGHTS]

        # update local model weights with received weights
        self.model.load_state_dict(
            {k: torch.as_tensor(v) for k, v in model_weights.items()}
        )

        self.local_train()

        # build the shareable
        shareable = Shareable()
        shareable[ShareableKey.META] = {FLConstants.NUM_STEPS_CURRENT_ROUND: 1}
        shareable[ShareableKey.TYPE] = ShareableValue.TYPE_WEIGHTS
        shareable[ShareableKey.DATA_TYPE] = ShareableValue.DATA_TYPE_UNENCRYPTED
        shareable[ShareableKey.MODEL_WEIGHTS] = {
            k: v.cpu().numpy() for k, v in self.model.state_dict().items()
        }

        self.logger.info("Local epochs finished.  Returning shareable")
        return shareable
