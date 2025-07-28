//
//  ETTrainerExecutor.swift
//  NVFlare iOS SDK
//
//  ExecutorTorch executor - handles both direct usage and config-based creation
//

import Foundation

/// ExecutorTorch implementation of NVFlareExecutor
/// Supports both direct instantiation and config-based creation via ComponentCreator
/// Clean architecture: Swift coordination → Direct C++ ExecutorTorch integration
public class ETTrainerExecutor: NSObject, NVFlareExecutor, ComponentCreator {
    private let trainerArgs: [String: Any]
    
    // MARK: - ComponentCreator (Resolver)
    
    /// Creates ETTrainerExecutor instances from config
    public static func create(name: String, args: [String: Any]) -> Any {
        return ETTrainerExecutor(args: args)
    }
    
    // MARK: - Initializers
    
    /// Default initializer for direct usage
    public override init() {
        self.trainerArgs = [:]
        super.init()
    }
    
    /// Config-based initializer with stored args
    public init(args: [String: Any]) {
        self.trainerArgs = args
        super.init()
    }
    
    // MARK: - NVFlareExecutor
    
    public func execute(taskData: NVFlareDXO, ctx: NVFlareContext, abortSignal: NVFlareSignal) -> NVFlareDXO {
        print("ETTrainerExecutor: Starting execute() with ExecutorTorch")
        
        // Extract model data from DXO
        guard let modelData = taskData.data["model"] as? String else {
            print("ETTrainerExecutor: No model data found in task data")
            return NVFlareDXO(dataKind: "error", data: ["error": "No model data found"])
        }
        
        // Merge stored args with task data meta (task data takes precedence)
        var finalConfig = trainerArgs
        for (key, value) in taskData.meta {
            finalConfig[key] = value
        }
        
        // Get training parameters from merged config
        let epochs = finalConfig["num_epochs"] as? Int ?? 1
        let batchSize = finalConfig["batch_size"] as? Int ?? 32
        let learningRate = finalConfig["learning_rate"] as? Float ?? 0.01
        
        // Get C++ dataset from FlareRunner context (set during FlareRunner initialization)
        guard let cppDatasetPtr = ctx[NVFlareContextKey.dataset] as? UnsafeMutableRawPointer else {
            print("❌ ETTrainerExecutor: No C++ dataset found in context")
            return NVFlareDXO(dataKind: "error", data: ["error": "No C++ dataset found in context"])
        }
        
        print("🔍 ETTrainerExecutor: Retrieved dataset pointer from context: \(cppDatasetPtr)")
        print("📊 ETTrainerExecutor: Training config - epochs: \(epochs), batch: \(batchSize), lr: \(learningRate)")
        if !trainerArgs.isEmpty {
            print("📊 ETTrainerExecutor: Using config args: \(trainerArgs.keys.sorted())")
        }
        print("📊 ETTrainerExecutor: Using app's C++ dataset directly")
        
        // Create training configuration dictionary for C++ layer
        let trainingConfig: [String: Any] = [
            "num_epochs": epochs,
            "batch_size": batchSize,
            "learning_rate": learningRate,
            "method": finalConfig["method"] as? String ?? "cnn",
            "dataset_shuffle": finalConfig["dataset_shuffle"] as? Bool ?? true
        ]
        
        // Direct approach - pass C++ dataset directly to ETTrainer
        print("ETTrainerExecutor: Passing C++ dataset directly to ETTrainer")
        
        // Create ETTrainer with app's C++ dataset
        let trainer = ETTrainer(
            modelBase64: modelData,
            meta: trainingConfig,
            dataset: cppDatasetPtr  // ← App's C++ ETDataset* directly!
        )
        
        // Handle genuine initialization failures (corrupt model, dataset issues, etc.)
        guard let trainer = trainer else {
            print("ETTrainerExecutor: ETTrainer initialization failed - check model data, dataset type, and ExecutorTorch dependencies")
            return NVFlareDXO(dataKind: "error", data: [
                "error": "ETTrainer initialization failed",
                "details": "Possible causes: invalid model data, unsupported dataset type, or ExecutorTorch loading issues",
                "config_args": trainerArgs
            ])
        }
        
        // Execute training (C++ ExecutorTorch does the heavy lifting)
        print("ETTrainerExecutor: Starting ExecutorTorch training...")
        let result = trainer.train()
        
        print("ETTrainerExecutor: Training completed with \(result.keys.count) result keys")
        
        // Convert result to DXO format
        return NVFlareDXO(
            dataKind: "model",
            data: result,
            meta: [
                "training_completed": true,
                "timestamp": Date().timeIntervalSince1970,
                "epochs": epochs,
                "learning_rate": learningRate,
                "batch_size": batchSize,
                "dataset": "app_cpp_dataset",
                "architecture": "executorch_native",
                "config_args": trainerArgs.isEmpty ? "none" : trainerArgs.keys.sorted()
            ]
        )
    }
} 
