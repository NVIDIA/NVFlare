//
//  SwiftDataset.swift
//  NVFlare iOS SDK
//
//  Swift protocol for app developers to provide training data
//

import Foundation

/// Standardized dataset errors for consistent error handling across all dataset implementations
public enum DatasetError: Error, LocalizedError {
    case noDataFound
    case invalidDataFormat
    case dataLoadFailed
    case emptyDataset
    case invalidConfiguration
    case unsupportedOperation
    
    public var errorDescription: String? {
        switch self {
        case .noDataFound:
            return "Dataset data not found"
        case .invalidDataFormat:
            return "Dataset data format is invalid"
        case .dataLoadFailed:
            return "Failed to load dataset data"
        case .emptyDataset:
            return "Dataset is empty"
        case .invalidConfiguration:
            return "Dataset configuration is invalid"
        case .unsupportedOperation:
            return "Operation not supported by this dataset"
        }
    }
    
    public var failureReason: String? {
        switch self {
        case .noDataFound:
            return "The required data files are missing from the app bundle or specified location"
        case .invalidDataFormat:
            return "The data file format does not match the expected structure"
        case .dataLoadFailed:
            return "An error occurred while loading the dataset data"
        case .emptyDataset:
            return "The dataset contains no valid samples"
        case .invalidConfiguration:
            return "The dataset configuration parameters are invalid"
        case .unsupportedOperation:
            return "The requested operation is not supported by this dataset type"
        }
    }
    
    public var recoverySuggestion: String? {
        switch self {
        case .noDataFound:
            return "Ensure the required data files are included in the app bundle or check the data path"
        case .invalidDataFormat:
            return "Verify the data file format matches the expected structure for this dataset"
        case .dataLoadFailed:
            return "Check file permissions and ensure the data files are accessible"
        case .emptyDataset:
            return "Verify the dataset contains valid data or check filtering criteria"
        case .invalidConfiguration:
            return "Review and correct the dataset configuration parameters"
        case .unsupportedOperation:
            return "Check the dataset documentation for supported operations"
        }
    }
}

/// Protocol for app developers to provide training data
/// Implement this protocol to create custom datasets for federated learning
public protocol SwiftDataset {
    /// Get a batch of training data
    /// - Parameter size: Number of samples to return
    /// - Returns: Array containing [inputsArray, labelsArray] or nil if no more data
    func getBatch(size: Int) -> [Any]?
    
    /// Reset the dataset iterator to the beginning
    func reset()
    
    /// Get the total number of samples in the dataset
    func size() -> Int
    
    /// Get the input dimension (number of features per sample)
    func inputDim() -> Int
    
    /// Get the label dimension (number of classes)
    func labelDim() -> Int
    
    /// Enable or disable data shuffling
    /// - Parameter shuffle: Whether to shuffle the data
    func setShuffle(_ shuffle: Bool)
}

/// Default implementations for common dataset operations
public extension SwiftDataset {
    /// Default implementation for setShuffle (no-op)
    func setShuffle(_ shuffle: Bool) {
        // Default implementation does nothing
        // Override in subclasses if shuffling is needed
    }
    
    /// Check if the dataset is empty
    var isEmpty: Bool {
        return size() == 0
    }
    
    /// Validate that the dataset has valid data
    /// - Throws: DatasetError.emptyDataset if the dataset is empty
    func validate() throws {
        if isEmpty {
            throw DatasetError.emptyDataset
        }
    }
} 
