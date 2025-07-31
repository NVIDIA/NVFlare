# ExecuTorch Build Comparison: With vs Without Training Support

## Overview
This document compares building ExecuTorch for Android with and without training support, highlighting key differences, requirements, and common pitfalls.

## Build Requirements

### System Requirements
- macOS or Linux (Windows not officially supported)
- Minimum 8GB RAM (16GB recommended)
- At least 10GB free disk space
- Python 3.8 or higher

### Common Requirements

#### Android NDK
- Version: 29.0.13113456
- Installation:
  ```bash
  # Using Android Studio
  1. Open Android Studio
  2. Go to Tools > SDK Manager
  3. Select "SDK Tools" tab
  4. Check "NDK (Side by side)"
  5. Click "Show Package Details"
  6. Select version 29.0.13113456
  7. Click "Apply" to install

  # Manual installation
  mkdir -p $HOME/Library/Android/sdk/ndk
  cd $HOME/Library/Android/sdk/ndk
  wget https://dl.google.com/android/repository/android-ndk-r29-linux-x86_64.zip
  unzip android-ndk-r29-linux-x86_64.zip
  mv android-ndk-r29 29.0.13113456
  ```

#### Android SDK
- Version: Latest stable (API 34 recommended)
- Installation:
  ```bash
  # Using Android Studio
  1. Download Android Studio from https://developer.android.com/studio
  2. Run the installer
  3. During setup, select "Custom" installation
  4. Ensure "Android SDK" is selected
  5. Complete the installation

  # Manual installation
  mkdir -p $HOME/Library/Android/sdk
  cd $HOME/Library/Android/sdk
  wget https://dl.google.com/android/repository/commandlinetools-mac-10406996_latest.zip
  unzip commandlinetools-mac-10406996_latest.zip
  ```

#### CMake
- Version: 3.22.1 or higher
- Installation:
  ```bash
  # macOS
  brew install cmake

  # Ubuntu/Debian
  sudo apt-get update
  sudo apt-get install cmake

  # Verify installation
  cmake --version
  ```

#### Python Dependencies
- Python 3.8+ with virtual environment
- Required packages (from executorch/requirements-dev.txt):
  ```
  numpy>=1.21.0
  pyyaml>=5.4.1
  requests>=2.25.1
  typing-extensions>=4.0.0
  ```
- Installation:
  ```bash
  # Create virtual environment
  python3 -m venv venv
  source venv/bin/activate

  # Install dependencies
  pip install -r executorch/requirements-dev.txt
  pip install -r executorch/requirements-examples.txt
  ```

#### Gradle
- Version: 8.0 or higher
- Installation:
  ```bash
  # macOS
  brew install gradle

  # Ubuntu/Debian
  sudo apt-get update
  sudo apt-get install gradle

  # Verify installation
  gradle --version
  ```

### Additional Requirements for Training

#### Training Extension Headers
- Located in: `executorch/extension/training/`
- Required files:
  - `module/training_module.h`
  - `optimizer/sgd.h`
  - `state_dict_util.h`

#### Optimizer Support
- Built-in optimizers:
  - SGD (Stochastic Gradient Descent)
  - Additional optimizers can be added by implementing the optimizer interface

#### Memory Allocator
- Required for training operations
- Configured via CMake flags:
  - `EXECUTORCH_BUILD_EXTENSION_TENSOR=ON`
  - `EXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON`

#### Data Loader
- Required for training data management
- Configured via CMake flag:
  - `EXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON`

## Environment Setup

### Required Environment Variables
```bash
# Android SDK and NDK paths
export ANDROID_HOME=$HOME/Library/Android/sdk
export ANDROID_NDK=$HOME/Library/Android/sdk/ndk/29.0.13113456
export PATH=$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools

# ExecuTorch specific
export EXECUTORCH_ROOT=$(pwd)/executorch
export ANDROID_ABIS=arm64-v8a
export EXECUTORCH_CMAKE_BUILD_TYPE=Release
```

### Verification Script
```bash
#!/bin/bash
# verify_dependencies.sh

# Check Android NDK
if [ ! -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake" ]; then
    echo "Error: NDK toolchain file not found"
    exit 1
fi

# Check CMake version
cmake_version=$(cmake --version | head -n1 | cut -d" " -f3)
if [ "$(echo "$cmake_version 3.22.1" | awk '{print ($1 >= $2)}')" -eq 0 ]; then
    echo "Error: CMake version must be 3.22.1 or higher"
    exit 1
fi

# Check Python version
python_version=$(python3 --version | cut -d" " -f2)
if [ "$(echo "$python_version 3.8.0" | awk '{print ($1 >= $2)}')" -eq 0 ]; then
    echo "Error: Python version must be 3.8.0 or higher"
    exit 1
fi

# Check Gradle version
gradle_version=$(gradle --version | grep Gradle | cut -d" " -f2)
if [ "$(echo "$gradle_version 8.0.0" | awk '{print ($1 >= $2)}')" -eq 0 ]; then
    echo "Error: Gradle version must be 8.0.0 or higher"
    exit 1
fi

echo "All dependencies verified successfully!"
```

## Build Process Differences

### Without Training
```bash
# Basic CMake configuration
cmake -B cmake-out-android-arm64-v8a \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android-arm64-v8a \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-26 \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_STL=c++_shared \
    -DBUILD_TESTING=OFF \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_XNNPACK_SHARED_WORKSPACE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
    -DEXECUTORCH_BUILD_ANDROID_JNI=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_LOG_LEVEL=Info
```

### With Training
```bash
# Extended CMake configuration
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
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TRAINING=ON \
    -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
    -DEXECUTORCH_BUILD_ANDROID_JNI=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DEXECUTORCH_LOG_LEVEL=Info \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON
```

## Key Differences

### 1. CMake Configuration
- Training build requires additional flags:
  - `EXECUTORCH_BUILD_TRAINING=ON`
  - `EXECUTORCH_BUILD_EXTENSION_TENSOR=ON`
  - `EXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON`
  - `EXECUTORCH_BUILD_EXTENSION_TRAINING=ON`
  - Additional kernel build options for training operations

### 2. Dependencies
- Training build includes:
  - Training module headers
  - Optimizer implementations
  - State dictionary utilities
  - Additional memory management for training

### 3. Output Files
- Both builds produce:
  - `executorch.aar`
  - Native libraries
- Training build additionally includes:
  - Training module headers
  - Optimizer headers
  - Training-specific JNI bindings

## Common Pitfalls

### 1. Missing Dependencies
- **Issue**: Build fails due to missing training components
- **Solution**: Ensure all training-related CMake flags are set

### 2. Header Path Issues
- **Issue**: Training headers not found during build
- **Solution**: Verify header paths and include directories

### 3. Memory Management
- **Issue**: Insufficient memory allocation for training operations
- **Solution**: Configure proper memory allocators

### 4. JNI Layer Integration
- **Issue**: Missing JNI bindings for training operations
- **Solution**: Ensure `EXECUTORCH_BUILD_ANDROID_JNI=ON` is set

## Best Practices

1. **Pre-build Verification**
   - Check all required components exist
   - Verify paths and dependencies
   - Ensure proper environment setup

2. **Build Process**
   - Use clean build directories
   - Verify CMake configuration
   - Check build logs for warnings

3. **Post-build Verification**
   - Verify AAR contents
   - Check header installation
   - Test JNI bindings

## Resource Usage

### Without Training
- Build time: ~5-10 minutes
- AAR size: ~2-3MB
- Memory usage: Moderate

### With Training
- Build time: ~10-15 minutes
- AAR size: ~3-4MB
- Memory usage: Higher due to training components

## Android STL Options

### DANDROID_STL=c++_shared
- **Purpose**: Specifies the C++ Standard Library implementation to use
- **Why c++_shared**:
  - Shared library version of the C++ standard library
  - Reduces APK size by sharing the library across multiple native libraries
  - Required for JNI applications that need to share C++ runtime
  - Better compatibility with other native libraries

### Alternative Options
- `c++_static`: Static linking of C++ standard library
  - Larger APK size but no runtime dependencies
  - Not recommended for JNI applications
- `none`: No C++ standard library
  - Minimal but limited functionality
  - Not suitable for complex C++ applications

### Impact on Build
- **AAR Size**: Affects the size of the final AAR file
- **Runtime Dependencies**: Determines if libc++_shared.so needs to be included
- **Compatibility**: Affects compatibility with other native libraries
- **Memory Usage**: Shared library reduces memory footprint when multiple native libraries are used

## Additional Configuration Options

### Installation Directory
- `CMAKE_INSTALL_PREFIX`: Specifies where built artifacts are installed
- Default: `cmake-out-android-arm64-v8a`
- Important for finding built libraries and headers

### Logging Configuration
- `EXECUTORCH_ENABLE_LOGGING`: Enables debug logging
- `EXECUTORCH_LOG_LEVEL`: Sets logging verbosity
  - Options: Debug, Info, Warning, Error
  - Default: Info
- Useful for debugging build and runtime issues

## Extension Dependencies

### Training Extension Dependencies
The training extension (`extension_training`) requires several other extensions to function properly:

1. **Core Dependencies** (from training/CMakeLists.txt):
   - `executorch_core`: Core runtime library
   - `extension_data_loader`: For loading training data
   - `extension_module_static`: For module management
   - `extension_tensor`: For tensor operations
   - `extension_flat_tensor`: For flat tensor representation

2. **Required CMake Flags**:
   ```cmake
   -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON
   -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON
   -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON
   -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON
   -DEXECUTORCH_BUILD_EXTENSION_TRAINING=ON
   ```

### Extension Relationships
- **Data Loader Extension**: Required for training data management
- **Module Extension**: Provides module loading and management
- **Tensor Extension**: Core tensor operations
- **Flat Tensor Extension**: Efficient tensor representation
- **Training Extension**: Builds on top of all above extensions

### Build Process Impact
1. **Dependency Order**: Extensions must be built in the correct order
2. **Memory Usage**: Each extension adds to the final binary size
3. **Build Time**: More extensions = longer build time

## Conclusion
Building ExecuTorch with training support requires additional configuration and dependencies but provides the necessary components for on-device training. The key is proper setup and verification to avoid resource-intensive build failures. 