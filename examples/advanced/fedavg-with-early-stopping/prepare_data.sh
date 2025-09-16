DATASET_ROOT="/tmp/nvflare/data/cifar10"

python3 -c "import torchvision.datasets as datasets; datasets.CIFAR10(root='${DATASET_ROOT}', train=True, download=True)"
