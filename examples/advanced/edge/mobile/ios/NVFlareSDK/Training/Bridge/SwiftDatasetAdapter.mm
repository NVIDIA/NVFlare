//
//  SwiftDatasetAdapter.mm
//  NVFlareSDK
//
//  C++ adapter implementation to bridge Swift datasets to ETDataset interface
//

#import "SwiftDatasetAdapter.h"
#include <executorch/extension/tensor/tensor.h>
#include <algorithm>
#include <stdexcept>
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>

// Selector names for Swift methods with @objc annotations
// These will be initialized at runtime using sel_registerName

using namespace executorch::extension;

SwiftDatasetAdapter::SwiftDatasetAdapter(void* swiftObject) 
    : swiftObjectPtr(swiftObject), isDestroyed(false) {
    NSLog(@"SwiftDatasetAdapter constructor: created at %p with Swift object %p", this, swiftObject);
}

SwiftDatasetAdapter::~SwiftDatasetAdapter() {
    NSLog(@"SwiftDatasetAdapter destructor called for %p", this);
    
    // Thread-safe check: atomically try to set isDestroyed from false to true
    bool expected = false;
    if (!isDestroyed.compare_exchange_strong(expected, true)) {
        // Another thread already destroyed this object
        NSLog(@"SwiftDatasetAdapter: Already destroyed by another thread! (expected=%s)", expected ? "true" : "false");
        return;
    }
    
    NSLog(@"SwiftDatasetAdapter: Successfully acquired destruction lock, proceeding with cleanup");
    
    if (swiftObjectPtr) {
        NSLog(@"SwiftDatasetAdapter: About to release retained Swift object at %p", swiftObjectPtr);
        
        @try {
            void* ptrToRelease = swiftObjectPtr;
            swiftObjectPtr = nullptr;
            
            NSLog(@"SwiftDatasetAdapter: Calling CFBridgingRelease on %p", ptrToRelease);
            CFBridgingRelease(ptrToRelease);
            NSLog(@"SwiftDatasetAdapter: CFBridgingRelease completed successfully");
            
        } @catch (NSException *exception) {
            NSLog(@"SwiftDatasetAdapter: Exception during CFBridgingRelease: %@", exception);
        }
        
        NSLog(@"SwiftDatasetAdapter: Swift object pointer cleared");
    } else {
        NSLog(@"SwiftDatasetAdapter: No retained Swift object to release");
    }
    
    NSLog(@"SwiftDatasetAdapter destructor completed successfully for %p", this);
}

std::optional<SwiftDatasetAdapter::BatchType> SwiftDatasetAdapter::getBatch(size_t batchSize) {
    NSLog(@"SwiftDatasetAdapter::getBatch() called with size %zu", batchSize);
    
    if (isDestroyed.load() || !swiftObjectPtr) {
        NSLog(@"SwiftDatasetAdapter::getBatch() object is destroyed or null");
        return std::nullopt;
    }
    
    @autoreleasepool {
        id nvflareDataset = (__bridge id)swiftObjectPtr;
        if (!nvflareDataset) {
            NSLog(@"SwiftDatasetAdapter::getBatch() bridged object is nil");
            return std::nullopt;
        }
        
        SEL getNextBatchSelector = sel_registerName("getNextBatchWithBatchSize:");
        if (![nvflareDataset respondsToSelector:getNextBatchSelector]) {
            NSLog(@"SwiftDatasetAdapter::getBatch() object does not respond to getNextBatchWithBatchSize:");
            return std::nullopt;
        }
        
        // Get NVFlareBatch from NVFlareDataset
        typedef id (*GetNextBatchFunc)(id, SEL, NSInteger);
        GetNextBatchFunc getNextBatchFunc = (GetNextBatchFunc)objc_msgSend;
        
        id nvflareBatch = getNextBatchFunc(nvflareDataset, getNextBatchSelector, (NSInteger)batchSize);
        if (!nvflareBatch) {
            NSLog(@"SwiftDatasetAdapter::getBatch() no more batches available");
            return std::nullopt;
        }
        
        // Extract input and label from NVFlareBatch
        SEL getInputSelector = sel_registerName("getInput");
        SEL getLabelSelector = sel_registerName("getLabel");
        if (![nvflareBatch respondsToSelector:getInputSelector] || 
            ![nvflareBatch respondsToSelector:getLabelSelector]) {
            NSLog(@"SwiftDatasetAdapter::getBatch() batch does not respond to getInput/getLabel");
            return std::nullopt;
        }
        
        id inputData = ((id (*)(id, SEL))objc_msgSend)(nvflareBatch, getInputSelector);
        id labelData = ((id (*)(id, SEL))objc_msgSend)(nvflareBatch, getLabelSelector);
        
        if (!inputData || !labelData) {
            NSLog(@"SwiftDatasetAdapter::getBatch() null input or label data from batch");
            return std::nullopt;
        }
        
        // Convert to arrays
        NSArray<NSNumber *> *inputs = (NSArray<NSNumber *> *)inputData;
        NSArray<NSNumber *> *labels = (NSArray<NSNumber *> *)labelData;
        
        if (![inputs isKindOfClass:[NSArray class]] || ![labels isKindOfClass:[NSArray class]]) {
            NSLog(@"SwiftDatasetAdapter::getBatch() input/label data is not NSArray");
            return std::nullopt;
        }
        
        std::vector<float> inputVec;
        std::vector<int64_t> labelVec;
        
        inputVec.reserve(inputs.count);
        labelVec.reserve(labels.count);
        
        for (NSNumber *input in inputs) {
            inputVec.push_back(input.floatValue);
        }
        
        for (NSNumber *label in labels) {
            labelVec.push_back(label.intValue);
        }
        
        if (labelVec.empty()) {
            NSLog(@"SwiftDatasetAdapter::getBatch() no labels in batch");
            return std::nullopt;
        }
        
        // Calculate input dimensions
        size_t batchSizeActual = labelVec.size();
        if (batchSizeActual == 0 || inputVec.size() % batchSizeActual != 0) {
            NSLog(@"SwiftDatasetAdapter::getBatch() input size %zu not evenly divisible by batch size %zu", inputVec.size(), batchSizeActual);
            return std::nullopt;
        }
        size_t inputDim = inputVec.size() / batchSizeActual;
        
        // Create tensors
        auto inputTensor = make_tensor_ptr<float>(
            {static_cast<int>(batchSizeActual), static_cast<int>(inputDim)}, 
            std::move(inputVec)
        );
        
        auto labelTensor = make_tensor_ptr<int64_t>(
            {static_cast<int>(batchSizeActual)}, 
            std::move(labelVec)
        );
        
        NSLog(@"SwiftDatasetAdapter::getBatch() returning batch with %zu samples", batchSizeActual);
        return std::make_pair(inputTensor, labelTensor);
    }
}

void SwiftDatasetAdapter::reset() {
    NSLog(@"SwiftDatasetAdapter::reset() called");
    
    if (isDestroyed.load() || !swiftObjectPtr) {
        NSLog(@"SwiftDatasetAdapter::reset() object is destroyed or null");
        return;
    }
    
    @autoreleasepool {
        id nvflareDataset = (__bridge id)swiftObjectPtr;
        if (!nvflareDataset) {
            NSLog(@"SwiftDatasetAdapter::reset() bridged object is nil");
            return;
        }
        
        SEL resetSelector = sel_registerName("reset");
        if ([nvflareDataset respondsToSelector:resetSelector]) {
            typedef void (*ResetFunc)(id, SEL);
            ResetFunc resetFunc = (ResetFunc)objc_msgSend;
            resetFunc(nvflareDataset, resetSelector);
            NSLog(@"SwiftDatasetAdapter::reset() completed");
        } else {
            NSLog(@"SwiftDatasetAdapter::reset() object does not respond to reset");
        }
    }
}

void SwiftDatasetAdapter::setShuffle(bool shuffle) {
    NSLog(@"SwiftDatasetAdapter::setShuffle() called with %s", shuffle ? "true" : "false");
    
    // NVFlareDataset protocol doesn't include setShuffle - this is a no-op
    // Shuffling should be handled internally by the dataset implementation
    NSLog(@"SwiftDatasetAdapter::setShuffle() NVFlareDataset protocol doesn't support setShuffle - no-op");
}

size_t SwiftDatasetAdapter::size() const {
    NSLog(@"SwiftDatasetAdapter::size() called on %p", this);
    
    if (isDestroyed.load()) {
        NSLog(@"SwiftDatasetAdapter::size() called on destroyed object!");
        return 0;
    }
    
    if (!swiftObjectPtr) {
        NSLog(@"SwiftDatasetAdapter::size() NVFlareDataset object pointer is null!");
        return 0;
    }
    
    NSLog(@"SwiftDatasetAdapter::size() calling NVFlareDataset object at %p", swiftObjectPtr);
    
    @autoreleasepool {
        id nvflareDataset = (__bridge id)swiftObjectPtr;
        if (!nvflareDataset) {
            NSLog(@"SwiftDatasetAdapter::size() bridged object is nil!");
            return 0;
        }
        
        SEL sizeSelector = sel_registerName("size");
        if (![nvflareDataset respondsToSelector:sizeSelector]) {
            NSLog(@"SwiftDatasetAdapter::size() object does not respond to size selector");
            return 0;
        }
        
        NSLog(@"SwiftDatasetAdapter::size() calling size method on valid object");
        
        // Safer casting for objc_msgSend
        typedef NSInteger (*SizeFunc)(id, SEL);
        SizeFunc sizeFunc = (SizeFunc)objc_msgSend;
        
        NSInteger result = sizeFunc(nvflareDataset, sizeSelector);
        NSLog(@"SwiftDatasetAdapter::size() result: %ld", (long)result);
        return static_cast<size_t>(result);
    }
}

size_t SwiftDatasetAdapter::inputDim() const {
    NSLog(@"SwiftDatasetAdapter::inputDim() called");
    
    if (isDestroyed.load() || !swiftObjectPtr) {
        NSLog(@"SwiftDatasetAdapter::inputDim() object is destroyed or null");
        return 0;
    }
    
    @autoreleasepool {
        id nvflareDataset = (__bridge id)swiftObjectPtr;
        if (!nvflareDataset) {
            NSLog(@"SwiftDatasetAdapter::inputDim() bridged object is nil");
            return 0;
        }
        
        SEL getInputDimensionsSelector = sel_registerName("getInputDimensions");
        if ([nvflareDataset respondsToSelector:getInputDimensionsSelector]) {
            typedef NSArray* (*GetInputDimensionsFunc)(id, SEL);
            GetInputDimensionsFunc getInputDimensionsFunc = (GetInputDimensionsFunc)objc_msgSend;
            NSArray *dimensions = getInputDimensionsFunc(nvflareDataset, getInputDimensionsSelector);
            if (dimensions && dimensions.count > 0) {
                NSInteger firstDim = [[dimensions firstObject] integerValue];
                NSLog(@"SwiftDatasetAdapter::inputDim() result: %ld", (long)firstDim);
                return static_cast<size_t>(firstDim);
            } else {
                NSLog(@"SwiftDatasetAdapter::inputDim() empty or null dimensions array");
                return 0;
            }
        } else {
            NSLog(@"SwiftDatasetAdapter::inputDim() ERROR: object does not respond to getInputDimensions - this is required!");
            throw std::runtime_error("Dataset must implement getInputDimensions method - cannot determine input dimensions");
        }
    }
}

size_t SwiftDatasetAdapter::labelDim() const {
    NSLog(@"SwiftDatasetAdapter::labelDim() called");
    
    if (isDestroyed.load() || !swiftObjectPtr) {
        NSLog(@"SwiftDatasetAdapter::labelDim() object is destroyed or null");
        return 1; // Default to single label
    }
    
    @autoreleasepool {
        id nvflareDataset = (__bridge id)swiftObjectPtr;
        if (!nvflareDataset) {
            NSLog(@"SwiftDatasetAdapter::labelDim() bridged object is nil");
            return 1;
        }
        
        SEL getOutputDimensionsSelector = sel_registerName("getOutputDimensions");
        if ([nvflareDataset respondsToSelector:getOutputDimensionsSelector]) {
            typedef NSArray* (*GetOutputDimensionsFunc)(id, SEL);
            GetOutputDimensionsFunc getOutputDimensionsFunc = (GetOutputDimensionsFunc)objc_msgSend;
            NSArray *dimensions = getOutputDimensionsFunc(nvflareDataset, getOutputDimensionsSelector);
            if (dimensions && dimensions.count > 0) {
                NSInteger firstDim = [[dimensions firstObject] integerValue];
                NSLog(@"SwiftDatasetAdapter::labelDim() result: %ld", (long)firstDim);
                return static_cast<size_t>(firstDim);
            } else {
                NSLog(@"SwiftDatasetAdapter::labelDim() empty or null dimensions array");
                return 1;
            }
        } else {
            NSLog(@"SwiftDatasetAdapter::labelDim() ERROR: object does not respond to getOutputDimensions - this is required!");
            throw std::runtime_error("Dataset must implement getOutputDimensions method - cannot determine label dimensions");
        }
    }
} 
