# NVFlareMobile

NVFlareMobile is a mobile implementation of NVIDIA FLARE (Federated Learning Application Runtime Environment) for iOS devices. This project enables on-device federated learning training using ExecutorTorch.

## Requirements

### System Requirements
- iOS 14.0+
- Xcode 14.0+
- Swift 5.0+
- C++17 or later

### Dependencies
- ExecutorTorch Framework
- SwiftUI for UI components

### Start development
Clone the repo, install Xcode on your platform, then open the [NVFlareMobile](./NVFlareMobile/) project inside Xcode.

You need to build the Executorch framework on your computer as well. Please refer to [Executorch Apple Runtime](https://pytorch.org/executorch/stable/apple-runtime.html).

Include those frameworks in the app and click the build and run button.

The app will be installed and started on your selected destination (iPhone or iPhone simulator).

## Project Structure

### Sources

#### App
- `ContentView.swift`: Main UI interface for the application

#### Connection
- `Connection.swift`: Handles network communication with FLARE server

#### Models
- **Common**
  - `NVFlareError.swift`: Error definitions for the application
- **Connection**
  - `JobResponse.swift`: Data models for server job responses
- **Training**
  - `Job.swift`: Job configuration and management
  - `TrainingTask.swift`: Training task definitions and parameters

#### Trainers
- **ETTrainer**
  - `ETTrainer.mm`: Core training implementation using ExecutorTorch
    - Supports model loading from base64 encoded strings
    - Implements tensor operations and parameter management
    - Provides training loop with SGD optimizer
    - Handles tensor serialization and deserialization
    - Supports parameter difference calculation for efficient updates

#### Training
- `Trainer.swift`: Base trainer interface
- `TrainerController.swift`: Manages training execution and lifecycle
- `DeviceStateMonitor.swift`: Monitors device conditions for training

## Features

1. **Model Management**
   - Load models from base64 encoded strings
   - Support for PTE (Portable Tensor Export) format
   - Parameter serialization and deserialization

2. **Training Capabilities**
   - Local training with configurable epochs
   - SGD optimizer implementation
   - Support for multiple local epochs
   - Parameter difference calculation for efficient updates

3. **Device Monitoring**
   - Device state tracking
   - Resource usage monitoring
   - Training condition verification

4. **Network Communication**
   - Server communication for model updates
   - Job management and coordination
   - Secure data transfer

## Implementation Notes

1. The implementation uses a hybrid approach with:
   - Swift for high-level application logic
   - Objective-C++ for ExecutorTorch integration
   - SwiftUI for modern UI implementation

2. Training features:
   - Supports multiple local epochs
   - Implements parameter difference calculation instead of raw gradients
   - Provides tensor manipulation utilities
   - Includes debugging and logging capabilities

3. Security considerations:
   - Secure model transfer using base64 encoding
   - Temporary file handling for model loading
   - Clean-up procedures for sensitive data

## Future Enhancements

1. Cross-platform support:
   - Android implementation
   - Browser-based training
   - Edge/wearable device support

2. Extended ML capabilities:
   - Support for additional model types
   - XGBoost integration
   - Enhanced optimization methods

3. Runtime improvements:
   - Investigation of LiteRT
   - ONNX runtime integration
   - Performance optimizations

## Contributing

Please refer to the contribution guidelines before submitting pull requests.


## User Guide & Development Workflow

### Getting Started

1. **Environment Setup**
   - Install Xcode 14.0 or later
   - Install ExecutorTorch dependencies
   - Clone the repository

2. **Building the Project**
   - Open `NVFlareMobile.xcodeproj` in Xcode
   - Select your target device/simulator
   - Build the project (⌘B)


### Troubleshooting

1. **Common Issues**
   - Build errors
   - Runtime crashes
   - Memory warnings
   - Network timeouts

2. **Solutions**
   - Clean and rebuild project
   - Check device logs
   - Verify network connectivity
   - Validate model format

3. **Support**
   - File issues on GitHub
   - Check documentation
   - Contact development team

### Deployment

1. **Release Checklist**
   - Run all tests
   - Check memory usage
   - Verify error handling
   - Test on multiple devices

2. **Distribution**
   - Archive build
   - Sign with certificates
   - Submit to App Store
   - Update documentation

## Core Architecture

### Component Responsibilities

1. **ContentView**
   - Main UI entry point
   - User interaction handling
   - Training status display

2. **TrainerController**
   - Central coordinator
   - Manages training lifecycle
   - Handles component interactions
   - State management

3. **DeviceMonitor**
   - Device state tracking
   - Resource monitoring
   - Training conditions check
   - Battery management

4. **Trainer (Protocol)**
   - Base training interface
   - Common training operations
   - Model management API

5. **ETTrainer**
   - ExecutorTorch implementation
   - Tensor operations
   - Training loop
   - Parameter management

6. **Connection**
   - Server communication
   - Model sync
   - Job management
   - Update handling

7. **TrainingTask**
   - Task configuration
   - Job metadata
   - Training parameters

### Data Flow
```ascii
                [ContentView] <-----> [TrainerController] <--> [DeviceMonitor]
                                           ^    ^
                                           |    |
                                           v    |
                                     [Connection] |
                                           ^     |
                                           v     |
                                       [Server]  |
                                                |
                                                v
                                           [Trainer]
```

### Key Interactions

1. **User Interface Flow**
   ```ascii
   ContentView (User Actions) <--> TrainerController <--> Connection
   ```

2. **Server Communication Flow**
   ```ascii
   TrainerController <--> Connection <--> Server
   ```

3. **Training Flow**
   ```ascii
   TrainerController <--> Trainer
   ```
