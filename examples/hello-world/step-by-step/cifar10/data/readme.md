# Problem and Data

For this tutorial, we will use the CIFAR10 dataset. 
It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. 
The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

The pytorch tutorial will training an image classifier. The examples shows the following steps in order:

* Load and normalize the CIFAR10 training and test datasets using torchvision
* Define a Convolutional Neural Network
* Define a loss function
* Train the network on the training data
* Test the network on the test data

![image](./cifar10.png)


# PREPARE DATA 

Before we start the working on the problem, let's first prepare the data, assuming the dataset is already available, 
we usually need two steps to prepare data.

* download the data

To avoid each job to download and split the data separately, we add a step to prepare the data for all the cifar10 jobs. 

The CIFAR10 data will be downloaded to the common location, so rest of the job won't download it.

to download data
```
cd cifar10/data
```

```python

python download.py

```

* Split the data so each client will have a portion of the data

Next step is to Split the data. In general there are different ways of splitting data, choose which way depends on the
type of problem and type of data. For simplicity, we assume all clients will have the same data for horizontal Federated learning cases.  

so nothing to do for data split, we just point all clients to the same data location 








