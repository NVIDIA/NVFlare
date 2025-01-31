# Problem and Data

For this tutorial, we will use the CIFAR-10 dataset. 
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. 
The images in CIFAR-10 are of size 3x32x32, i.e., 3-channel color images with a resolution of 32x32 pixels.

The PyTorch tutorial will train an image classifier. The example shows the following steps in order:

* Load and normalize the CIFAR10 training and test datasets using torchvision
* Define a Convolutional Neural Network
* Define a loss function
* Train the network on the training data
* Test the network on the test data

![image](./cifar10.png)


# Prepare Data

Before we start the working on the problem, let's first prepare the data, assuming the dataset is already available, 
we usually need two steps to prepare data.

* download the data

To avoid each job having to download and split the data, we add a step to prepare the data for all the cifar10 jobs. 

The CIFAR-10 data will be downloaded to a common location, so it will not need to be repeatedly downloaded.

to download data
```
cd cifar10/data
```

```bash

python download.py

```

* Split the data so each client will have a portion of the data

In real-world scenarios, the data will be distributed among different clients/sides. We don't need this step. 

In simulated cases, we may need to split the data into different clients/sites. How to split the data, 
depending on the type of problem or type of data. For simplicity, in this example we assume all clients will have the same data for horizontal federated learning cases.
Thus we do not do a data split, but rather point all clients to the same data location.









