//
//  SwiftDataset.swift
//  NVFlare iOS SDK
//
//  Swift protocol for app developers to provide training data
//

import Foundation

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
} 
