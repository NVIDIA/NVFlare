# ExecuTorch Model Fine-Tuning on Android App

In this tutorial, we will be fine-tuning a CIFAR 10 model on an android app using ExecuTorch.

## Environment Setup

### Android Environment Setup

Ensure that your development environment meets the following requirements:

1. Minimum JDK version: 17
2. Target Android API level: 34 (`Android 14.0`)
3. Required Android SDK components:
   - Android SDK Build-Tools 34
   - Android SDK Platform-Tools
   - Android NDK (Side by side)
   - Android SDK Command-line Tools (latest)
4. Install Android Emulator or connect an Android device
5. Ensure the following environment variables are set:
   - `JAVA_HOME`
   - `ANDROID_NDK`
   - `ANDROID_SDK`

**NOTE** For the updated steps for building the dependencies refer to the official repository over [here](https://github.com/pytorch/executorch/blob/main/extension/android/README.md).

### ExecuTorch Environment Setup

To ensure better management of Python environments and packages, it is recommended to use a Python environment management tool such as `conda`, `venv`, or `uv`. For this demonstration, we will use `uv` to set up the Python environment.

To install ExecuTorch in a `uv` Python environment use the following commands:

```bash
$ git clone https://github.com/pytorch/executorch.git --recurse-submodules
$ cd executorch

# [optional] Setup an environment
$ uv venv --seed --prompt et --python 3.10
$ source .venv/bin/activate

# Install build tools and ExecuTorch pip wheel
$ ./install_executorch.sh

# Build ExecuTorch for Android. Output AAR: ./extension/android/executorch_android/build/outputs/aar/executorch_android-debug.aar
$ ./scripts/build_android_library.sh

# Fetch the examples
$ git clone git@github.com:pytorch-labs/executorch-examples.git

# Copy the ExecuTorch AAR into the libs directory
$ mkdir -p executorch-examples/cifar/android/CifarETTrainingDemo/app/libs
$ cp ./extension/android/executorch_android/build/outputs/aar/executorch_android-debug.aar executorch-examples/cifar/android/CifarETTrainingDemo/app/libs/executorch.aar

# Build the data and models
$ mkdir -p executorch-examples/cifar/android/CifarETTrainingDemo/app/src/main/assets/
$ python ./extension/training/examples/CIFAR/main.py --data-dir executorch-examples/cifar/android/CifarETTrainingDemo/app/src/main/assets/ --model-path executorch-examples/cifar/android/CifarETTrainingDemo/app/src/main/assets/cifar10_model.pth --pte-model-path executorch-examples/cifar/android/CifarETTrainingDemo/app/src/main/assets/cifar10_model.pte --split-pte-model-path executorch-examples/cifar/android/CifarETTrainingDemo/app/src/main/assets/cifar10_model_pte_only.pte --save-pt-json executorch-examples/cifar/android/CifarETTrainingDemo/app/src/main/assets/cifar10_pt.json --save-et-json executorch-examples/cifar/android/CifarETTrainingDemo/app/src/main/assets/cifar10_et.json --ptd-model-dir executorch-examples/cifar/android/CifarETTrainingDemo/app/src/main/assets/ --epochs 5 --fine-tune-epochs 10
```

If you run into errors for sdk path, complete the above steps in the [Trouble Shooting](#trouble-shooting) section before proceeding.

## Creation of Android App

1. Start with a clone of this repository and open the project in Android Studio executorch-examples/cifar/android/CifarETTrainingDemo/app/build.gradle.kts.

2. Set the minimum SDK version to `API 34`.

3. Wait for the Gradle sync to complete.

4. Click on run to proceed: ![](./images/Pasted%20image%2020250709170837.png)

5. Click on the fine-tune button and the training begins for `150` epochs![[Pasted image 20250709171210.png]]

    **Note:** The training parameters can be tweaked in `MainActivity.kt`

### Summary:

We trained the PyTorch model for `1 epoch` and exported the `.pte` and `.ptd` files. We started with training loss of `2.159132883071899`, training accuracy of `18%`, testing loss of `2.1072397136688235`, and testing accuracy of `29%`

```log
2025-07-09 17:11:46.309 13277-13277 ExecuTorchApp           com.example.democifar10              D  Starting Epoch 1/150
2025-07-09 17:11:46.309 13277-13277 ExecuTorchApp           com.example.democifar10              D  Total images to be trained: 500, Number of batches: 125
2025-07-09 17:11:47.237 13277-13428 EGL_emulation           com.example.democifar10              D  app_time_stats: avg=7.23ms min=2.75ms max=25.70ms count=60
2025-07-09 17:11:47.715 13277-13277 ExecuTorchApp           com.example.democifar10              D  Epoch [1/150] Loss: 2.159132883071899, Accuracy: 18%, Time: 1.41 s, Time per image: 2.81 ms
2025-07-09 17:11:47.715 13277-13277 ExecuTorchApp           com.example.democifar10              D  Starting evaluation
2025-07-09 17:11:47.715 13277-13277 ExecuTorchApp           com.example.democifar10              D  Evaluating model on 100 test images (25 batches) out of 100 total images
2025-07-09 17:11:47.989 13277-13277 ExecuTorchApp           com.example.democifar10              D  Evaluation complete - Loss: 2.1072397136688235, Accuracy: 29.0%, Time: 0.27 s, Time per image: 2.73 ms
```

We reached a training loss of `1.7837489886283875`, training accuracy of `35%`, testing loss of `2.0501016211509704`, and testing accuracy of `38%` after `150 epochs`

```log
2025-07-09 17:13:59.889 13277-13277 ExecuTorchApp           com.example.democifar10              D  Starting Epoch 150/150
2025-07-09 17:13:59.889 13277-13277 ExecuTorchApp           com.example.democifar10              D  Total images to be trained: 500, Number of batches: 125
2025-07-09 17:14:00.666 13277-13277 ExecuTorchApp           com.example.democifar10              D  Epoch [150/150] Loss: 1.7837489886283875, Accuracy: 35%, Time: 0.78 s, Time per image: 1.56 ms
2025-07-09 17:14:00.666 13277-13277 ExecuTorchApp           com.example.democifar10              D  Starting evaluation
2025-07-09 17:14:00.666 13277-13277 ExecuTorchApp           com.example.democifar10              D  Evaluating model on 100 test images (25 batches) out of 100 total images
2025-07-09 17:14:00.809 13277-13277 ExecuTorchApp           com.example.democifar10              D  Evaluation complete - Loss: 1.7236163129806519, Accuracy: 38%, Time: 0.67 s, Time per image: 1.35 ms
```

### Trouble Shooting

Check if the `local.properties` file is present in the `extension/android` directory:

```bash
$ cat ./extension/android/local.properties
```

**NOTE:** If this file (`local.properties`) is missing, the build process will fail because the script can't retrieve the path for the android sdk from the env variables. If you encounter this issue, please follow the following steps:

```bash
$ touch ./extension/android/local.properties
$ vim ./extension/android/local.properties
# Add the path to your sdk directory into this file like: sdk.dir=/Users/<USERNAME>/Library/Android/sdk
```
