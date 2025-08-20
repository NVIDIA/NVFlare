//
//  TrainingConstants.swift
//  NVFlare iOS SDK
//
//  Training-specific constants and dataset types
//

import Foundation

/// Training-specific constants for dataset types and training configurations
public struct TrainingConstants {
    
    // MARK: - Dataset Types
    public static let datasetTypeCIFAR10 = "cifar10"
    public static let datasetTypeXOR = "xor"
    
    // MARK: - Training Method Types
    public static let methodCNN = "cnn"
    public static let methodMLP = "mlp"
    
    // MARK: - Private initializer to prevent instantiation
    private init() {}
}
