DATASET_ROOT="/tmp/cifar10"

python -c "import torchvision.datasets as datasets; datasets.CIFAR10(root='${DATASET_ROOT}', train=True, download=True)"
