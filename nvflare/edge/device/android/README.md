# NVFlare Android App with ExecuTorch

This guide will help you set up and build the NVFlare Android application that uses ExecuTorch for federated learning on mobile devices.

## 🎯 What You'll Build

A complete Android application that:
- Runs federated learning using ExecuTorch
- Trains CIFAR-10 models on Android devices
- Integrates with the NVFlare federated learning framework
- Supports on-device training and model updates

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

### Required Software
- **Java Development Kit (JDK)**: OpenJDK 17 or later
- **Android SDK**: API level 29+ with latest build tools
- **Android NDK**: Version 29.0.13599879 or compatible
- **Python**: 3.10+ (for ExecuTorch build scripts)
- **Git**: For repository management
- **CMake**: For building native components
- **Xcode Command Line Tools** (macOS only): Required for build tools like `make`

### Quick Installation
```bash
# macOS with Homebrew
brew install openjdk@17
brew install cmake

# Install Xcode Command Line Tools (macOS only)
xcode-select --install
```

Install Android SDK/NDK via Android Studio SDK Manager
or download from: https://developer.android.com/studio

After installing Android Studio, install NDK
1. Open **Android Studio**
2. Go to **Tools → SDK Manager** (or **Android Studio → Settings → Appearance & Behavior → System Settings → Android SDK** on newer versions)
3. Click the **SDK Tools** tab
4. Check the box for **NDK (Side by side)**
5. Optionally, click **Show Package Details** to select version **29.0.13599879** specifically
6. Click **Apply** or **OK** to install


## 🚀 Quick Start

Follow these steps to get your NVFlare Android app running:

### Step 1: Set Up Environment Variables

```bash
# Set your Android development environment
export JAVA_HOME=/opt/homebrew/Cellar/openjdk@17/17.0.15/libexec/openjdk.jdk/Contents/Home
export ANDROID_HOME=/Users/$(whoami)/Library/Android/sdk
export ANDROID_NDK=/Users/$(whoami)/Library/Android/sdk/ndk/29.0.13599879
export ANDROID_SDK=/Users/$(whoami)/Library/Android/sdk
export PATH=$PATH:$ANDROID_HOME/tools:$ANDROID_HOME/platform-tools
```

> **Note**: Adjust the paths above to match your system configuration.

### Step 2: Build ExecuTorch Libraries

```bash
# Create Python environment
python3.12 -m venv androidexecutorchenv
source androidexecutorchenv/bin/activate

# Clone and build ExecuTorch
git clone https://github.com/pytorch/executorch.git --recurse-submodules
cd executorch

# Update and install dependencies (clean if you have attempted installing previously)
git pull
./install_executorch.sh --clean
git submodule sync --recursive
git submodule update --init --recursive

# Install with training extensions
CMAKE_ARGS="-DEXECUTORCH_BUILD_EXTENSION_TRAINING=ON,-DEXECUTORCH_BUILD_PYBIND=ON" ./install_executorch.sh

# Build Android library
EXECUTORCH_BUILD_EXTENSION_LLM=OFF ./scripts/build_android_library.sh
```

### Step 3: Copy ExecuTorch Libraries

```bash
# Create libs directory
mkdir -p /path/to/NVFlare/nvflare/edge/device/android/app/libs

# Copy the ExecuTorch AAR file
cp ./extension/android/executorch_android/build/outputs/aar/executorch_android-debug.aar \
   /path/to/NVFlare/nvflare/edge/device/android/app/libs/executorch.aar
```

### Step 4: Set Up NVFlare SDK

**Critical**: The NVFlare Android SDK must be copied to the app's source directory.

```bash
# Copy SDK to app source directory
cp -r nvflare/edge/device/android/sdk \
      nvflare/edge/device/android/app/src/main/java/com/nvidia/nvflare/
```

### Step 5: Add Training Data

```bash
mkdir nvflare/edge/device/android/app/src/main/assets
# Copy CIFAR-10 dataset from the location in the repo used for iOS
cp nvflare/edge/device/ios/ExampleProject/ExampleApp/Assets.xcassets/cifar10/data_batch_1.dataset/data_batch_1.bin \
   nvflare/edge/device/android/app/src/main/assets/data_batch_1.bin
```

### Step 6: Build and Run

Open the project in Android Studio and build the app, or use the command line:

```bash
cd nvflare/edge/device/android
./gradlew assembleDebug
```

## 📁 Project Structure

```
nvflare/edge/device/android/
├── app/
│   ├── libs/                    # ExecuTorch libraries (.aar files)
│   ├── src/main/
│   │   ├── assets/              # Training data (CIFAR-10)
│   │   └── java/com/nvidia/nvflare/
│   │       ├── app/             # Main application code
│   │       └── sdk/             # NVFlare Android SDK (copied from sdk/)
├── sdk/                         # NVFlare Android SDK source
│   ├── core/                    # Core SDK functionality
│   ├── training/                # Training components
│   ├── utils/                   # Utility functions
│   └── models/                  # Model definitions
└── README.md                    # This file
```

## 🔧 Configuration

### Build Configuration

The `build.gradle.kts` file is pre-configured to automatically include all `.aar` files from the `libs/` directory:

```kotlin
implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("*.jar", "*.aar"))))
```

### Environment Setup

Make sure your `local.properties` file contains the correct Android SDK path:

```properties
sdk.dir=/Users/yourusername/Library/Android/sdk
```

## 🐛 Troubleshooting

### Common Issues

**CMake Error: "Unable to find a build program corresponding to Unix Makefiles" (macOS)**
- This occurs when Xcode Command Line Tools are not installed
- Solution:
  ```bash
  xcode-select --install
  ```
- Verify installation:
  ```bash
  which make  # Should output: /usr/bin/make
  ```
- If already installed but still having issues, try resetting:
  ```bash
  sudo xcode-select --reset
  ```

**Build fails with missing classes**
- Ensure the SDK has been copied to the app's source directory (Step 4)
- Verify all required `.aar` files are in the `libs/` directory

**ExecuTorch build fails**
- Check that all environment variables are set correctly
- Ensure you have sufficient disk space (build requires ~10GB)
- Try building with `EXECUTORCH_BUILD_EXTENSION_LLM=OFF` if LLM extensions cause issues

**App crashes on startup**
- Verify the CIFAR-10 dataset is in the correct location
- Check that all required permissions are granted in `AndroidManifest.xml`

### Getting Help

1. Check the [NVFlare documentation](https://nvflare.readthedocs.io/)
2. Review the [ExecuTorch Android examples](https://github.com/pytorch/executorch/tree/main/examples)
3. Open an issue in the NVFlare repository for Android-specific problems

## 📚 Additional Resources

- [NVFlare Documentation](https://nvflare.readthedocs.io/)
- [ExecuTorch Documentation](https://pytorch.org/executorch/)
- [Android Development Guide](https://developer.android.com/guide)
- [Federated Learning Concepts](https://nvflare.readthedocs.io/en/latest/fl_introduction.html)

## 🎉 Next Steps

Once your app is running:

1. **Test Training**: Run a simple training session to verify everything works
2. **Connect to Server**: Set up connection to an NVFlare server for federated learning
3. **Customize Models**: Modify the model architecture for your specific use case
4. **Deploy**: Package and deploy your app to Android devices

---

**Happy Federated Learning! 🚀**