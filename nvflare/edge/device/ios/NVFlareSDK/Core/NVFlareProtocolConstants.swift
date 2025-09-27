//
//  NVFlareProtocolConstants.swift
//  NVFlare iOS SDK
//
//  Standard NVFlare federated learning protocol constants
//  These are the agreed-upon meta keys between NVFlare server and clients
//

import Foundation

/// Standard NVFlare protocol constants for server-client communication
/// These meta keys are part of the official NVFlare protocol specification
public struct NVFlareProtocolConstants {
    
    // MARK: - Standard Meta Keys
    /// Batch size for training - standard NVFlare meta key
    public static let metaKeyBatchSize = "batch_size"
    
    /// Learning rate for training - standard NVFlare meta key
    public static let metaKeyLearningRate = "learning_rate"
    
    /// Total epochs for training - standard NVFlare meta key
    public static let metaKeyTotalEpochs = "total_epochs"
    
    /// Dataset shuffle flag - standard NVFlare meta key
    public static let metaKeyDatasetShuffle = "dataset_shuffle"
    
    /// Dataset type identifier - standard NVFlare meta key
    public static let metaKeyDatasetType = "dataset_type"
    
    // MARK: - Private initializer to prevent instantiation
    private init() {}
}
