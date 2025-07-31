#!/bin/bash

# Exit on error
set -e

# Force Python to run unbuffered
export PYTHONUNBUFFERED=1

# Set up environment variables with hardcoded paths
export ANDROID_HOME=/Users/kevlu/Library/Android/sdk
export ANDROID_NDK=/Users/kevlu/Library/Android/sdk/ndk/29.0.13113456
export PATH=$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools
export EXECUTORCH_ROOT=$(pwd)/executorch
export ANDROID_ABIS=arm64-v8a
export EXECUTORCH_CMAKE_BUILD_TYPE=Release

# Verify NDK toolchain file exists
if [ ! -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ]; then
    echo "Error: NDK toolchain file not found at $ANDROID_NDK/build/cmake/android.toolchain.cmake"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check and install required dependencies
echo "Checking required dependencies..."
REQUIRED_PACKAGES=(
    "numpy>=1.21.0"
    "pyyaml>=5.4.1"
    "requests>=2.25.1"
    "typing-extensions>=4.0.0"
)

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! pip show "${package%>=*}" >/dev/null 2>&1; then
        echo "Installing $package..."
        pip install "$package"
    else
        echo "✓ $package is already installed"
    fi
done

# Install dependencies
pip install -r executorch/requirements-dev.txt

# Clean up existing build directory
rm -rf executorch/build
# Clean up existing build directories
rm -rf executorch/cmake-out-android-*
rm -rf executorch/cmake-out-android-so

# Change to executorch directory
cd executorch

# Configure CMake with all required options for training
cmake -B cmake-out-android-arm64-v8a \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android-arm64-v8a \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-26 \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_STL=c++_shared \
    -DBUILD_TESTING=OFF \
    -DEXECUTORCH_BUILD_TRAINING=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TRAINING=ON \
    -DEXECUTORCH_BUILD_ANDROID_JNI=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_LOG_LEVEL=Info \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DEXECUTORCH_BUILD_PYTHON_BINDINGS=OFF \
    -DEXECUTORCH_BUILD_TRAINING_PYBIND=OFF

# Build the core library with verbose output
cmake --build cmake-out-android-arm64-v8a -j4 --target install --config Release --verbose

# Create jni/include directory in AAR
mkdir -p extension/android/executorch_android/src/main/jni/include/executorch/extension/training/module
mkdir -p extension/android/executorch_android/src/main/jni/include/executorch/extension/training/optimizer
mkdir -p extension/android/executorch_android/src/main/jni/include/executorch/extension/module

# Copy headers to AAR
cp extension/training/module/*.h extension/android/executorch_android/src/main/jni/include/executorch/extension/training/module/
cp extension/training/optimizer/*.h extension/android/executorch_android/src/main/jni/include/executorch/extension/training/optimizer/
cp extension/module/*.h extension/android/executorch_android/src/main/jni/include/executorch/extension/module/

# Build the Android AAR
export ANDROID_SDK=/Users/kevlu/Library/Android/sdk
./scripts/build_android_library.sh

# Create app libs directory
mkdir -p ../appforexecutorchtraining/app/libs

# Copy the AAR to our app's libs directory
cp extension/android/executorch_android/build/outputs/aar/executorch_android-debug.aar ../appforexecutorchtraining/app/libs/executorch.aar

# Create include directories
mkdir -p ../appforexecutorchtraining/app/libs/include/executorch/extension/training/module
mkdir -p ../appforexecutorchtraining/app/libs/include/executorch/extension/training/optimizer
mkdir -p ../appforexecutorchtraining/app/libs/include/executorch/extension/data_loader
mkdir -p ../appforexecutorchtraining/app/libs/include/executorch/extension/flat_tensor/serialize
mkdir -p ../appforexecutorchtraining/app/libs/include/executorch/extension/tensor

# Copy all required headers
cp extension/training/module/*.h ../appforexecutorchtraining/app/libs/include/executorch/extension/training/module/
cp extension/training/optimizer/*.h ../appforexecutorchtraining/app/libs/include/executorch/extension/training/optimizer/
cp extension/data_loader/*.h ../appforexecutorchtraining/app/libs/include/executorch/extension/data_loader/
cp extension/flat_tensor/serialize/*.h ../appforexecutorchtraining/app/libs/include/executorch/extension/flat_tensor/serialize/
cp extension/tensor/*.h ../appforexecutorchtraining/app/libs/include/executorch/extension/tensor/

# Verify the headers were copied
echo "Verifying headers..."
ls -l ../appforexecutorchtraining/app/libs/include/executorch/extension/training/module/
ls -l ../appforexecutorchtraining/app/libs/include/executorch/extension/training/optimizer/
ls -l ../appforexecutorchtraining/app/libs/include/executorch/extension/module/

echo "Build completed successfully!" 