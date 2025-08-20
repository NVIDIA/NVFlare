//
//  NVFlareConstants.swift
//  NVFlare iOS SDK
//
//  Core SDK constants and configuration keys
//

import Foundation

/// Core SDK constants for configuration and metadata
public struct NVFlareConstants {
    
    // MARK: - Meta Keys
    public static let metaKeyBatchSize = "batch_size"
    public static let metaKeyLearningRate = "learning_rate"
    public static let metaKeyTotalEpochs = "total_epochs"
    public static let metaKeyDatasetShuffle = "dataset_shuffle"
    public static let metaKeyDatasetType = "dataset_type"
    
    // MARK: - Private initializer to prevent instantiation
    private init() {}
}
