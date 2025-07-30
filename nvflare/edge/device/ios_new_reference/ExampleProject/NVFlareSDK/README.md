# NVFlareSDK for iOS

[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![iOS](https://img.shields.io/badge/iOS-14.0+-blue.svg)](https://developer.apple.com/ios/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

NVFlareSDK enables iOS apps to participate in federated learning using NVIDIA FLARE. Train models locally on iOS devices while preserving user privacy.

## 🚀 Quick Start

### 1. Add NVFlareSDK to Your Project

#### Swift Package Manager (Recommended)

In Xcode, go to **File → Add Package Dependencies** and add:

```
https://github.com/NVIDIA/NVFlare-iOS-SDK
```

Or add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/NVIDIA/NVFlare-iOS-SDK", from: "1.0.0")
]
```

### 2. Basic Integration (Mock Training)

```swift
import SwiftUI
import NVFlareSDK

struct ContentView: View {
    @StateObject private var flareRunner = NVFlareRunner(
        serverHost: "your-flare-server.com",
        serverPort: 4321
    )
    
    var body: some View {
        VStack {
            Button("Start Federated Learning") {
                Task {
                    await flareRunner.run()
                }
            }
            
            Button("Stop Training") {
                flareRunner.stop()
            }
        }
    }
}
```

## 🔧 Production Integration (Real Training)

For production use, integrate ExecutorTorch for real on-device training:

### 1. Add ExecutorTorch to Your Project

Download ExecutorTorch XCFrameworks and add to your project:
- `executorch.xcframework`
- `backend_xnnpack.xcframework` (recommended for iOS)

### 2. Create Your ETTrainer Implementation

```objc
// RealETTrainer.h
#import <Foundation/Foundation.h>
#import <NVFlareSDK/NVFlareSDK-Swift.h>

@interface RealETTrainer : NSObject <ETTrainerProtocol>
- (instancetype)initWithModelBase64:(NSString *)modelBase64 
                               meta:(NSDictionary<NSString *, id> *)meta;
- (NSDictionary<NSString *, id> *)train;
@end
```

```objc
// RealETTrainer.mm
#import "RealETTrainer.h"
#import <executorch/executorch.h>

@implementation RealETTrainer {
    std::unique_ptr<torch::executorch::Module> _module;
    NSDictionary *_meta;
}

- (instancetype)initWithModelBase64:(NSString *)modelBase64 
                               meta:(NSDictionary<NSString *, id> *)meta {
    if (self = [super init]) {
        _meta = meta;
        // Load ExecutorTorch model from base64
        NSData *modelData = [[NSData alloc] initWithBase64EncodedString:modelBase64 options:0];
        // Initialize ExecutorTorch module...
    }
    return self;
}

- (NSDictionary<NSString *, id> *)train {
    // Implement real training with ExecutorTorch
    // Return weight differences
    return @{
        @"model_diff": @{/* actual weight differences */},
        @"loss": @(/* actual loss */),
        @"accuracy": @(/* actual accuracy */)
    };
}

@end
```

### 3. Configure Your App to Use Real Training

```swift
// AppDelegate.swift or App.swift
import NVFlareSDK

@main
struct MyApp: App {
    init() {
        // Configure NVFlareSDK to use your real ExecutorTorch implementation
        ETTrainerFactory.setTrainerType(RealETTrainer.self)
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
```

## 📱 Features

### Core Capabilities
- ✅ **Federated Learning Client** - Complete FL client implementation
- ✅ **Server Communication** - Secure communication with FLARE servers
- ✅ **Component System** - Extensible component architecture
- ✅ **Data Privacy** - All training happens on-device
- ✅ **Production Ready** - Battle-tested in real deployments

### Supported Training Backends
- 🔬 **Mock Training** - Built-in for testing and development
- 🚀 **ExecutorTorch** - Production ML training on iOS
- 🔌 **Extensible** - Bring your own training implementation

### Device Support
- 📱 **iOS 14.0+** - iPhone and iPad
- 🖥️ **iOS Simulator** - For development and testing
- ⚡ **Apple Silicon** - Optimized for M1/M2 devices

## 🛠️ Advanced Usage

### Custom Data Sources

```swift
class MyDataSource: NVFlareDataSource {
    func getDataset(datasetType: String, ctx: NVFlareContext) -> NVFlareDataset? {
        switch datasetType {
        case "my_dataset":
            return MyCustomDataset()
        default:
            return nil
        }
    }
}

// Use custom data source
let runner = NVFlareRunner(
    serverHost: "server.com",
    serverPort: 4321,
    customDataSource: MyDataSource()
)
```

### Custom Filters

```swift
class PrivacyFilter: NSObject, NVFlareFilter {
    func filter(data: NVFlareDXO, ctx: NVFlareContext, abortSignal: NVFlareSignal) -> NVFlareDXO {
        // Apply differential privacy, encryption, etc.
        return data
    }
}

// Add to runner
let runner = NVFlareRunner(
    serverHost: "server.com", 
    serverPort: 4321,
    inFilters: [PrivacyFilter()],
    outFilters: [PrivacyFilter()]
)
```

## 📋 Requirements

### Minimum Requirements
- iOS 14.0+ / iPadOS 14.0+
- Xcode 14.0+
- Swift 5.9+

### Production Requirements (for real training)
- ExecutorTorch XCFrameworks
- C++17 compiler support
- Minimum 2GB RAM recommended
- Metal support for GPU acceleration (optional)

## 📖 Documentation

- 📘 **[Integration Guide](docs/integration.md)** - Detailed setup instructions
- 🔧 **[ExecutorTorch Setup](docs/executorch-setup.md)** - Step-by-step ExecutorTorch integration
- 📊 **[Examples](examples/)** - Sample projects and use cases
- 🔍 **[API Reference](docs/api-reference.md)** - Complete API documentation

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📚 **Documentation**: [NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/NVIDIA/NVFlare/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/NVIDIA/NVFlare/issues)

---

**Made with ❤️ by NVIDIA** 