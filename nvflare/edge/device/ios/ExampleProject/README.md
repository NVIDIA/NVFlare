# NVFlareMobile Demo App

A demonstration iOS app showing how to integrate federated learning using ExecutorTorch and NVFlareSDK.

## Overview

This is a simple SwiftUI demo app that shows how to:
- ✅ **Integrate ExecutorTorch**: Use pre-built ExecutorTorch frameworks
- ✅ **Include NVFlareSDK**: Copy the SDK folder into your app
- ✅ **Simple FL Training**: Start federated learning with CIFAR-10 or XOR datasets
- ✅ **Real-time Status**: Monitor training progress and status

## Prerequisites for App Developers

### 1. ExecutorTorch Framework
```bash
# Clone ExecutorTorch
git clone https://github.com/pytorch/executorch.git
cd executorch

# Build Apple framework
./build_apple_framework.sh

# Copy the built framework to your project's "Frameworks" folder
cp -r build/apple_framework/* /path/to/your/app/Frameworks/
```

### 2. NVFlareSDK
```bash
# Copy the NVFlareSDK folder into your app
cp -r nvflare/edge/device/ios/NVFlareSDK nvflare/edge/device/ios/ExampleProject
```

## Project Structure

```
ExampleProject/
├── Frameworks/              # ExecutorTorch frameworks go here
│   ├── ExecutorTorch.xcframework
│   └── ...
├── NVFlareSDK/              # Copied from nvflare/edge/device/ios/NVFlareSDK
│   ├── Core/
│   ├── Models/
│   └── ...
├── ExampleApp/              # App code
│   ├── ContentView.swift    # UI
│   ├── TrainerController    # Use NVFlareRunner to enable FL training
│   └── Datasets             # Implement NVFlareDataset interface to feed your own device data
```

## App Features

### Server Configuration
- Set NVFlare server hostname and port
- Default: `192.168.6.101:4321`

### Training Methods
- **CIFAR-10**: Image classification with CNN
- **XOR**: Simple neural network training
- Toggle which methods your app supports

### Training Control
- Start/stop federated learning
- Real-time status updates
- Error handling and recovery

