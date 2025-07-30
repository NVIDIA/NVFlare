//
//  SwiftCIFAR10Dataset.swift
//  ExampleApp
//
//  Swift implementation of CIFAR10 dataset
//

import Foundation
import UIKit

/// Swift implementation of CIFAR10 dataset
public class SwiftCIFAR10Dataset: SwiftDataset {
    private struct CIFARImage {
        let label: Int
        let data: [Float]
    }
    
    private let images: [CIFARImage]
    private var indices: [Int]
    private var currentIndex: Int = 0
    private var shouldShuffle: Bool = false
    
    // CIFAR-10 constants
    private let imageWidth = 32
    private let imageHeight = 32
    private let channels = 3
    private let imageSize = 32 * 32 * 3 // 3072 bytes per image
    private let maxImages = 16 // Demo limit
    
    public init(shuffle: Bool = false) throws {
        self.images = try Self.loadCIFAR10Data()
        self.indices = Array(0..<images.count)
        self.shouldShuffle = shuffle
        reset()
    }
    
    public func getBatch(size: Int) -> [Any]? {
        if currentIndex >= images.count {
            return nil
        }
        
        let endIndex = min(currentIndex + size, images.count)
        let actualBatchSize = endIndex - currentIndex
        
        var inputs: [NSNumber] = []
        var labels: [NSNumber] = []
        
        inputs.reserveCapacity(actualBatchSize * imageSize)
        labels.reserveCapacity(actualBatchSize)
        
        for i in currentIndex..<endIndex {
            let image = images[indices[i]]
            inputs.append(contentsOf: image.data.map { NSNumber(value: $0) })
            labels.append(NSNumber(value: image.label))
        }
        
        currentIndex = endIndex
        return [NSArray(array: inputs), NSArray(array: labels)]
    }
    
    @objc public func reset() {
        currentIndex = 0
        if shouldShuffle {
            indices.shuffle()
        } else {
            indices = Array(0..<images.count)
        }
    }
    
    @objc public func size() -> Int {
        return images.count
    }
    
    @objc public func inputDim() -> Int {
        return imageSize
    }
    
    @objc public func labelDim() -> Int {
        return 1
    }
    
    @objc public func setShuffle(_ shuffle: Bool) {
        shouldShuffle = shuffle
        reset()
    }
    
    // MARK: - Private Methods
    
    private static func loadCIFAR10Data() throws -> [CIFARImage] {
        var images: [CIFARImage] = []
        
        // Try to load CIFAR-10 data from app bundle
        guard let dataAsset = NSDataAsset(name: "data_batch_1") else {
            print("SwiftCIFAR10Dataset: No data_batch_1 found in app bundle.")
            throw DatasetError.noDataFound
        }
        
        let binaryData = dataAsset.data
        let bytes = [UInt8](binaryData)
        
        // Validate data format
        let bytesPerImage = 1 + 3072 // 1 byte label + 3072 bytes image data
        guard bytes.count >= bytesPerImage else {
            print("SwiftCIFAR10Dataset: Data file too small. Expected at least \(bytesPerImage) bytes, got \(bytes.count)")
            throw DatasetError.invalidDataFormat
        }
        
        let numImages = min(bytes.count / bytesPerImage, 16) // Demo limit
        
        guard numImages > 0 else {
            print("SwiftCIFAR10Dataset: No valid images found in data file")
            throw DatasetError.emptyDataset
        }
        
        for i in 0..<numImages {
            let startIndex = i * bytesPerImage
            let label = Int(bytes[startIndex])
            
            var imageData: [Float] = []
            imageData.reserveCapacity(3072)
            
            // Convert raw bytes to normalized float values [0,1]
            for j in 1..<bytesPerImage {
                let pixelValue = Float(bytes[startIndex + j]) / 255.0
                imageData.append(pixelValue)
            }
            
            images.append(CIFARImage(label: label, data: imageData))
        }
        
        print("SwiftCIFAR10Dataset: Successfully loaded \(images.count) CIFAR-10 images from app bundle")
        return images
    }
} 
