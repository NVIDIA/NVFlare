//
//  ETTrainerComponentResolver.swift
//  NVFlare iOS SDK
//
//  Specialized component resolver for ETTrainer that handles server parameter mappings
//

import Foundation

/// Specialized resolver for ETTrainer components
/// Handles server-to-iOS parameter mappings specifically for training components
public class ETTrainerComponentResolver: ComponentCreator {
    
    /// Create ETTrainerExecutor from server configuration
    /// Handles parameter name mappings from server format to iOS format
    public static func create(name: String, args: [String: Any]) -> Any {
        let mappedArgs = mapServerArgsToiOSFormat(args)
        return ETTrainerExecutor(args: mappedArgs)
    }
    
    /// Maps server parameter names to iOS format
    /// This encapsulates the mapping logic that was previously hardcoded in JobResponse
    private static func mapServerArgsToiOSFormat(_ serverArgs: [String: Any]) -> [String: Any] {
        var mappedArgs: [String: Any] = [:]
        
        for (key, value) in serverArgs {
            switch key {
            // Training parameters
            case "epoch":
                mappedArgs["num_epochs"] = value
                
            case "lr":
                if let doubleVal = value as? Double {
                    mappedArgs[NVFlareProtocolConstants.metaKeyLearningRate] = Float(doubleVal)
                } else if let floatVal = value as? Float {
                    mappedArgs[NVFlareProtocolConstants.metaKeyLearningRate] = floatVal
                }
                
            case "batch_size":
                mappedArgs[NVFlareProtocolConstants.metaKeyBatchSize] = value
                
            case "shuffle":
                mappedArgs[NVFlareProtocolConstants.metaKeyDatasetShuffle] = value
                
            // Pass through standard protocol keys unchanged
            case NVFlareProtocolConstants.metaKeyLearningRate,
                 NVFlareProtocolConstants.metaKeyBatchSize,
                 NVFlareProtocolConstants.metaKeyTotalEpochs,
                 NVFlareProtocolConstants.metaKeyDatasetShuffle,
                 NVFlareProtocolConstants.metaKeyDatasetType:
                mappedArgs[key] = value
                
            // Pass through other parameters unchanged
            default:
                mappedArgs[key] = value
            }
        }
        
        return mappedArgs
    }
}

/// Extension to register ETTrainer with enhanced resolver
extension ComponentRegistry {
    
    /// Register ETTrainer with specialized resolver instead of basic ETTrainerExecutor
    public func registerETTrainerWithResolver() {
        register(serverType: "Trainer.DLTrainer", creator: ETTrainerComponentResolver.self)
    }
}
