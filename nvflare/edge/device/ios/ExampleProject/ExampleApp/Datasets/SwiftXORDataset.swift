//
//  SwiftXORDataset.swift
//  ExampleApp
//
//  Swift implementation of XOR dataset
//

import Foundation

/// Swift implementation of XOR dataset
public class SwiftXORDataset: NSObject, NVFlareDataset {
    private let xorTable: [(inputs: [Float], label: Int)] = [
        ([1.0, 1.0], 0),
        ([0.0, 0.0], 0),
        ([1.0, 0.0], 1),
        ([0.0, 1.0], 1)
    ]
    
    private var indices: [Int]
    private var currentIndex: Int = 0
    private var shouldShuffle: Bool = false
    
    public init(shuffle: Bool = false) throws {
        self.indices = Array(0..<xorTable.count)
        self.shouldShuffle = shuffle
        super.init()
        reset()
    }
    
    @objc(getNextBatchWithBatchSize:) public func getNextBatch(batchSize: Int) -> NVFlareBatch {
        // Check if we've reached the end of the dataset
        if currentIndex >= xorTable.count {
            print("SwiftXORDataset: Reached end of dataset, resetting for next epoch")
            reset()
        }
        
        // Calculate actual batch size (may be smaller for the last batch)
        let remainingSamples = xorTable.count - currentIndex
        let actualBatchSize = min(batchSize, remainingSamples)
        
        if actualBatchSize == 0 {
            return NVFlareDataBatch(input: NSArray(), label: NSArray(), batchSize: 0)
        }
        
        let endIndex = currentIndex + actualBatchSize
        
        var inputs: [NSNumber] = []
        var labels: [NSNumber] = []
        
        inputs.reserveCapacity(actualBatchSize * 2) // 2 features per sample
        labels.reserveCapacity(actualBatchSize)
        
        for i in currentIndex..<endIndex {
            let data = xorTable[indices[i]]
            inputs.append(contentsOf: data.inputs.map { NSNumber(value: $0) })
            labels.append(NSNumber(value: data.label))
        }
        
        currentIndex = endIndex
        
        print("SwiftXORDataset: Returning batch with \(actualBatchSize) samples (requested: \(batchSize), remaining: \(xorTable.count - currentIndex))")
        
        return NVFlareDataBatch(
            input: NSArray(array: inputs),
            label: NSArray(array: labels),
            batchSize: actualBatchSize
        )
    }
    
    @objc public func reset() {
        currentIndex = 0
        if shouldShuffle {
            indices.shuffle()
        } else {
            indices = Array(0..<xorTable.count)
        }
    }
    
    @objc public func size() -> Int {
        return xorTable.count
    }
    
    @objc public func setShuffle(_ shuffle: Bool) {
        shouldShuffle = shuffle
        reset()
    }
    
    // MARK: - NVFlareDataset Protocol Methods
    
    @objc(getInputDimensions) public func getInputDimensions() -> [Int] {
        return [2] // XOR has 2 input features
    }
    
    @objc(getOutputDimensions) public func getOutputDimensions() -> [Int] {
        return [1] // XOR has 1 output (binary classification)
    }
} 
