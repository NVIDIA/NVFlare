#!/bin/bash

# Set up environment variables
export ANDROID_HOME=~/Library/Android/sdk
export ANDROID_NDK_HOME=~/Library/Android/sdk/ndk/29.0.13113456
export PATH=$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools
export EXECUTORCH_ROOT=$(pwd)/executorch

# Verify NDK toolchain file exists
if [ ! -f "$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" ]; then
    echo "Error: Android NDK toolchain file not found at $ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Build ExecuTorch with training support
cd executorch

# Configure CMake with all required options
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_TRAINING=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_ANDROID_JNI=ON \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-21 \
    -DANDROID_STL=c++_shared

# Build
cmake --build build -j4

# Build Android AAR with training support
python3 -m examples.portable.scripts.build_android_aar \
    --include_training \
    --include_optimizer \
    --include_module

# Copy the AAR to our app's libs directory
cp build/android/executorch.aar ../NVFlare/nvflare/edge/android/app/libs/

# Also copy the training module headers
mkdir -p ../NVFlare/nvflare/edge/android/app/src/main/cpp/include/training
cp extension/training/module/*.h ../NVFlare/nvflare/edge/android/app/src/main/cpp/include/training/
cp extension/training/optimizer/*.h ../NVFlare/nvflare/edge/android/app/src/main/cpp/include/training/ 