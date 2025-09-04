//
//  TrainingConstants.swift
//  NVFlare iOS SDK
//
//  iOS SDK-specific training implementation constants
//

import Foundation

/// iOS SDK-specific constants for training implementations, dataset types, and methods
/// Protocol constants (meta keys) are in NVFlareProtocolConstants.swift
public struct TrainingConstants {
    
    // MARK: - Dataset Types (iOS SDK-specific implementations)
    public static let datasetTypeCIFAR10 = "cifar10"
    public static let datasetTypeXOR = "xor"
    
    // MARK: - Training Method Types (iOS SDK-specific implementations)
    public static let methodCNN = "cnn"
    public static let methodMLP = "mlp"
    
    // MARK: - Private initializer to prevent instantiation
    private init() {}
}
