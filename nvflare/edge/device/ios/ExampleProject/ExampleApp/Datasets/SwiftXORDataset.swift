//
//  SwiftXORDataset.swift
//  ExampleApp
//
//  Swift implementation of XOR dataset
//

import Foundation

/// Swift implementation of XOR dataset
public class SwiftXORDataset: SwiftDataset {
    private let xorTable: [(inputs: [Float], label: Int)] = [
        ([1.0, 1.0], 0),
        ([0.0, 0.0], 0),
        ([1.0, 0.0], 1),
        ([0.0, 1.0], 1)
    ]
    
    private var indices: [Int]
    private var currentIndex: Int = 0
    private var shouldShuffle: Bool = false
    
    public init(shuffle: Bool = false) {
        self.indices = Array(0..<xorTable.count)
        self.shouldShuffle = shuffle
        reset()
    }
    
    public func getBatch(size: Int) -> [Any]? {
        if currentIndex >= xorTable.count {
            return nil
        }
        
        let endIndex = min(currentIndex + size, xorTable.count)
        let actualBatchSize = endIndex - currentIndex
        
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
        return [NSArray(array: inputs), NSArray(array: labels)]
    }
    
    public func reset() {
        currentIndex = 0
        if shouldShuffle {
            indices.shuffle()
        } else {
            indices = Array(0..<xorTable.count)
        }
    }
    
    public func size() -> Int {
        return xorTable.count
    }
    
    public func inputDim() -> Int {
        return 2
    }
    
    public func labelDim() -> Int {
        return 1
    }
    
    public func setShuffle(_ shuffle: Bool) {
        shouldShuffle = shuffle
        reset()
    }
} 