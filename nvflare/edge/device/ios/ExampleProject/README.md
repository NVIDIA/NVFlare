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
cp -r nvflare/edge/device/ios/ExampleProject/NVFlareSDK /path/to/your/app/
```

## Project Structure

```
ExampleApp/
├── Frameworks/           # ExecutorTorch frameworks go here
│   ├── ExecutorTorch.xcframework
│   └── ...
├── NVFlareSDK/          # Copied from this example
│   ├── Core/
│   ├── Executors/
│   └── ...
├── TrainerController.swift  # Main FL coordinator
├── ContentView.swift        # UI
└── ExampleApp.swift         # App entry point
```

## Features

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

## Usage

### Basic Setup
```swift
@StateObject private var trainer = TrainerController()

// Configure server
trainer.serverHost = "192.168.6.101"
trainer.serverPort = 4321

// Select training methods
trainer.supportedJobs = [.cifar10, .xor]
```

### Start Training
```swift
Button("Start Training") {
    Task {
        try await trainer.startTraining()
    }
}
```

### Monitor Status
```swift
switch trainer.status {
case .idle:
    Text("Ready to train")
case .training:
    Text("Training in progress...")
case .stopping:
    Text("Stopping...")
}
```

## Development

### Requirements
- iOS 14.0+
- Xcode 14.0+
- Swift 5.9+

### Building
1. Ensure ExecutorTorch frameworks are in `Frameworks/` folder
2. Verify NVFlareSDK folder is copied to your app
3. Open project in Xcode
4. Build and run (⌘R)

## Customization

### Add Your Own Dataset
```swift
// In Swift - just implement the SwiftDataset protocol
class MyCustomDataset: SwiftDataset {
    private let myData: [MyDataPoint]
    private var currentIndex = 0
    
    init(data: [MyDataPoint]) {
        self.myData = data
    }
    
    func getBatch(size: Int) -> [NSNumber]? {
        // Convert your data to training format
        let endIndex = min(currentIndex + size, myData.count)
        let batchData = myData[currentIndex..<endIndex]
        
        let inputs = batchData.flatMap { $0.features }.map { NSNumber(value: $0) }
        let labels = batchData.map { NSNumber(value: $0.label) }
        
        currentIndex = endIndex
        return [NSArray(array: inputs), NSArray(array: labels)]
    }
    
    func reset() { currentIndex = 0 }
    func size() -> Int { return myData.count }
    func inputDim() -> Int { return myData.first?.features.count ?? 0 }
    func labelDim() -> Int { return 1 }
    func setShuffle(_ shuffle: Bool) { /* implement if needed */ }
}

// Use it
let myDataset = MyCustomDataset(data: myData)
trainer.startTraining(with: myDataset)
```

### Modify Supported Jobs
```swift
enum SupportedJob: String, CaseIterable {
    case cifar10 = "CIFAR10"
    case xor = "XOR"
    case myCustom = "MY_CUSTOM"  // Add your own
    
    var displayName: String {
        switch self {
        case .cifar10: return "CIFAR-10"
        case .xor: return "XOR"
        case .myCustom: return "My Custom"
        }
    }
}
```

