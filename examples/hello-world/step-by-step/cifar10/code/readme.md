
# SAG workflow example with pytorch 

## Code example

the CIFAR10 example with pytorch is adapted from [pytorch tutorial example](https://github.com/pytorch/tutorials/blob/main/beginner_source/blitz/cifar10_tutorial.py)

we added the CLI parser to make it easier to handle CLI arguments. 

to run, in ``dl`` directory, we could run

```
python train.py

```

## Convert the example to FL

Now, in ```fl``` directory, we have converted this example to FL code using the FLARE Client Lightning API.

to run this code, we also need to specify the workflow and other configurations needed

to do this, we will leverage the FLARE's job CLI.


## Create job folder:

assuming the current directory is

```
examples/hello-world/step-by-step/cifar10/sag_pt
```

we can create a job folder

```bash
nvflare job create -j /tmp/nvflare/cifar10_sag -w sag_pt -s fl/train.py 
```

### Run job

* run job with simulator

```
    nvflare simulator /tmp/nvflare/cifar10_sag -w /tmp/cifar10_sag 
```

* Run in POC mode or Production

Before you can use POC or production mode, you must ensure that the server or clients are already started.
You can refer the POC setup tutorial to see how to setup the POC, and documentation to refer to the
production setup.

In POC mode, you can simply use the following CLI command to submit job.
```
    nvflare job submit -j /tmp/nvflare/cifar10_sag
```

In production, you will need to tell the CLI job command the location of the admin console startup kit directory

```
    nvflare config startup_kit_dir=<startup_kit_dir>
```
then

```
    nvflare job submit -j /tmp/nvflare/cifar10_sag
```



