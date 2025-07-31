# NVFlare Android Edge SDK Refactoring Documentation

## Overview

This document describes the refactoring of the NVFlare Android app from the old training-based architecture to the new Edge SDK architecture. The refactoring maintains backward compatibility while providing a more modular, extensible, and maintainable codebase.

## Architecture Comparison

### Previous Architecture (Old)

The previous implementation was built around a simple training loop with these key components:

```
TrainerController (ViewModel)
├── Connection (HTTP client)
├── Trainer (interface)
│   └── ETTrainerWrapper (implementation)
└── Models (JobResponse, TaskResponse, etc.)
```

**Key Characteristics:**
- Direct HTTP communication via `Connection` class
- Simple `Trainer` interface for model training
- Manual job/task fetching and result reporting
- Tightly coupled components
- Limited extensibility

### New Architecture (Edge SDK)

The new implementation introduces a modular SDK architecture:

```
FlareRunnerController (ViewModel) [Drop-in replacement]
├── AndroidFlareRunner (SDK implementation)
│   ├── Connection (reused from old architecture)
│   ├── DataSource (new abstraction)
│   ├── Executor (new abstraction)
│   └── Filters & EventHandlers (new features)
├── ConfigProcessor (dynamic configuration)
└── Component Registry (plugin system)
```

**Key Characteristics:**
- Abstract `FlareRunner` base class for platform independence
- Component-based configuration system
- Plugin architecture with component resolvers
- Standardized data exchange via DXO (Data Exchange Objects)
- Filter and event handler support
- Better separation of concerns

## File Structure and Implementation

### Core SDK Files

#### `/nvflare/edge/android/app/src/main/java/com/nvidia/nvflare/sdk/`

**`FlareRunner.kt`** - Abstract base class
- Main orchestrator for federated learning
- Handles job fetching, task execution, and result reporting
- Defines the core workflow and lifecycle
- Platform-agnostic design

**`AndroidFlareRunner.kt`** - Android-specific implementation
- Concrete implementation of `FlareRunner` for Android
- Bridges old `Connection` class with new SDK
- Implements the three required abstract methods:
  - `getJob()` - Fetches jobs from server
  - `getTask()` - Fetches tasks for current job
  - `reportResult()` - Sends training results back

**`FlareRunnerController.kt`** - Drop-in replacement for TrainerController
- Provides identical interface to old `TrainerController`
- Uses new `AndroidFlareRunner` internally
- Maintains all existing functionality (status tracking, method toggling)
- Enables seamless migration from old to new architecture

**`AndroidDataSource.kt`** - Data abstraction layer
- Implements `DataSource` interface for training data
- Provides datasets for different training methods (CIFAR-10, XOR)
- Extensible for new dataset types

**`AndroidExecutor.kt`** - Training execution abstraction
- Implements `Executor` interface for training tasks
- Bridges old `Trainer` interface with new SDK
- Factory pattern for creating different executor types

**`MigrationExample.kt`** - Migration documentation
- Examples showing how to migrate from old to new architecture
- Demonstrates backward compatibility
- Provides usage patterns

### SDK Definition Files

#### `/nvflare/edge/android/app/src/main/java/com/nvidia/nvflare/sdk/defs/`

**`Interfaces.kt`** - Core SDK interfaces
- `DataSource` - Provides training datasets
- `Executor` - Executes training tasks
- `Filter` - Transforms input/output data
- `EventHandler` - Responds to training events
- `Batch` & `Dataset` - Data handling abstractions

**`Context.kt`** - Context management
- Shared context for training execution
- Key-value store for passing data between components

**`DXO.kt`** - Data Exchange Objects
- Standardized format for data exchange
- Supports serialization/deserialization
- Type-safe data handling

**`Signal.kt`** - Abort signal mechanism
- Thread-safe abort signaling
- Enables graceful cancellation of operations

### Configuration System

#### `/nvflare/edge/android/app/src/main/java/com/nvidia/nvflare/sdk/config/`

**`ConfigProcessor.kt`** - Dynamic configuration processing
- Processes training configurations at runtime
- Component resolution and instantiation
- Plugin architecture support

### Supporting Infrastructure

#### `/nvflare/edge/android/app/src/main/java/com/nvidia/nvflare/models/`

**`JsonExtensions.kt`** - JSON utilities
- Extension methods for `JsonObject` and `JsonArray`
- Conversion to Kotlin collections
- Type-safe JSON handling

## Migration Path

### Backward Compatibility

The refactoring maintains **100% backward compatibility** with existing code:

```kotlin
// OLD CODE (still works)
val trainerController = ViewModelProvider(this)[TrainerController::class.java]
trainerController.status.observe(this) { status -> /* update UI */ }
trainerController.startTraining()

// NEW CODE (same interface!)
val flareRunnerController = ViewModelProvider(this)[FlareRunnerController::class.java]
flareRunnerController.status.observe(this) { status -> /* update UI */ }
flareRunnerController.startTraining()
```

### Migration Steps

1. **Immediate Migration** (Already Complete)
   - Replace `TrainerController` with `FlareRunnerController`
   - No changes to UI or business logic required
   - All existing functionality preserved

2. **Gradual Enhancement** (Future)
   - Add custom filters for data transformation
   - Implement event handlers for monitoring
   - Use dynamic configuration for training setup
   - Add custom executors for specialized training

3. **Full SDK Adoption** (Optional)
   - Direct use of `AndroidFlareRunner` for advanced use cases
   - Custom component implementations
   - Platform-specific optimizations

## Key Benefits

### 1. Modularity
- Components can be developed and tested independently
- Easy to add new training methods, datasets, and filters
- Clear separation of concerns

### 2. Extensibility
- Plugin architecture for custom components
- Dynamic configuration loading
- Support for custom training workflows

### 3. Maintainability
- Standardized interfaces and data formats
- Better error handling and logging
- Consistent patterns across the codebase

### 4. Platform Independence
- Abstract base classes enable cross-platform development
- Platform-specific implementations isolated
- Shared core logic across platforms

### 5. Backward Compatibility
- Existing code continues to work unchanged
- Gradual migration path available
- No breaking changes to public APIs

## Current Implementation Status

### ✅ Completed (Step 1)
- [x] `AndroidFlareRunner` - Core SDK implementation
- [x] `FlareRunnerController` - Drop-in replacement for TrainerController
- [x] `AndroidDataSource` - Basic data abstraction
- [x] `AndroidExecutor` - Training execution bridge (updated to use ETTrainerWrapper)
- [x] `JsonExtensions` - Supporting utilities
- [x] Migration examples and documentation

### 🔄 In Progress / Next Steps
- [ ] Real dataset implementations (CIFAR-10, XOR)
- [ ] Real training implementation in ETTrainer
- [ ] Custom filter implementations
- [ ] Event handler system
- [ ] Advanced configuration features

### 📋 Future Enhancements
- [ ] Custom component resolvers
- [ ] Performance optimizations
- [ ] Advanced monitoring and logging
- [ ] Security enhancements
- [ ] Cross-platform compatibility

## Usage Examples

### Basic Usage (Same as Old API)

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var flareRunnerController: FlareRunnerController
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Create controller (same as old TrainerController)
        flareRunnerController = ViewModelProvider(this)[FlareRunnerController::class.java]
        
        // Observe status (same interface)
        flareRunnerController.status.observe(this) { status ->
            when (status) {
                TrainingStatus.IDLE -> showIdleState()
                TrainingStatus.TRAINING -> showTrainingState()
                TrainingStatus.STOPPING -> showStoppingState()
            }
        }
        
        // Start training (same interface)
        binding.startButton.setOnClickListener {
            flareRunnerController.startTraining()
        }
        
        // Stop training (same interface)
        binding.stopButton.setOnClickListener {
            flareRunnerController.stopTraining()
        }
    }
}
```

### Advanced Usage (New SDK Features)

```kotlin
// Direct FlareRunner usage for advanced scenarios
val dataSource = AndroidDataSource()
val deviceInfo = mapOf("device_id" to "android_device_001")
val userInfo = mapOf("user_id" to "user123")

val flareRunner = AndroidFlareRunner(
    context = context,
    connection = connection,
    dataSource = dataSource,
    deviceInfo = deviceInfo,
    userInfo = userInfo,
    jobTimeout = 300.0f,
    resolverRegistry = customResolvers
)

// Run in background thread
lifecycleScope.launch(Dispatchers.IO) {
    flareRunner.run()
}
```

## File Mapping

| Old Component | New Component | Purpose |
|---------------|---------------|---------|
| `TrainerController` | `FlareRunnerController` | Main controller (drop-in replacement) |
| `Trainer` | `Executor` | Training execution abstraction |
| `Connection` | `Connection` (reused) | HTTP communication |
| `ETTrainerWrapper` | `AndroidExecutor` | Training implementation bridge |
| N/A | `AndroidFlareRunner` | Core SDK implementation |
| N/A | `AndroidDataSource` | Data abstraction layer |
| N/A | `ConfigProcessor` | Dynamic configuration |
| N/A | `DXO` | Standardized data format |

## Conclusion

The refactoring successfully introduces a modern, modular SDK architecture while maintaining complete backward compatibility. The new architecture provides:

1. **Immediate benefits** through better code organization and maintainability
2. **Future extensibility** through the plugin architecture
3. **Zero migration cost** for existing applications
4. **Clear upgrade path** for advanced use cases

The implementation demonstrates how to evolve a codebase without breaking existing functionality, enabling teams to adopt new features at their own pace while maintaining system stability. 