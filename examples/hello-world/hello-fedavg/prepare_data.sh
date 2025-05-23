DATASET_ROOT="~/data"

python3 -c "import torchvision.datasets as datasets; datasets.CIFAR10(root='${DATASET_ROOT}', train=True, download=True)"
