# ExecuTorch Libraries

This directory should contain the ExecuTorch Android libraries (.aar files) that are required for the NVFlare Android app.

## Required Libraries

Based on the CIFAR-10 ExecuTorch example, you need to add the following libraries:

1. **executorch.aar** - Main ExecuTorch library
2. **executorch_training.aar** - ExecuTorch training library (if separate)

## How to Obtain

1. **From ExecuTorch Release**: Download the latest ExecuTorch Android libraries from the official releases
2. **From CIFAR-10 Example**: Copy the libraries from the working CIFAR-10 example project
3. **Build from Source**: Build ExecuTorch from source and extract the Android libraries

## Installation Steps

1. Download the required .aar files
2. Place them in this `libs/` directory
3. The build.gradle.kts is already configured to include all .aar files from this directory

## Note

The build.gradle.kts file uses:
```kotlin
implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.jar", "*.aar"))))
```

This will automatically include all .jar and .aar files in this directory.

## Current Status

⚠️ **Libraries not yet added** - You need to add the ExecuTorch libraries before the app will compile and run successfully.

# Clone ExecuTorch repository
git clone https://github.com/pytorch/executorch.git --recurse-submodules
cd executorch

# Setup Python environment (optional but recommended)
uv venv --seed --prompt et --python 3.10
source .venv/bin/activate

# Install build tools and ExecuTorch pip wheel
./install_executorch.sh

# Build ExecuTorch for Android
./scripts/build_android_library.sh

# Create libs directory in NVFlare Android app
mkdir -p nvflare/edge/device/android/app/libs

# Copy the ExecuTorch AAR
cp ./extension/android/executorch_android/build/outputs/aar/executorch_android-debug.aar nvflare/edge/device/android/app/libs/executorch.aar

Copy nvflare/edge/ios/NVFlareMobile/NVFlareMobile/Assets.xcassets/cifar10/data_batch_1.dataset/data_batch_1.bin to nvflare/edge/device/android/app/src/main/assets/data_batch_1.bin

