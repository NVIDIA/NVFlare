from et_task_processor import ETTaskProcessor
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CIFAR10ETTaskProcessor(ETTaskProcessor):
    def get_dataset(self, data_path: str) -> Dataset:
        transform = transforms.Compose([transforms.ToTensor()])
        return datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
