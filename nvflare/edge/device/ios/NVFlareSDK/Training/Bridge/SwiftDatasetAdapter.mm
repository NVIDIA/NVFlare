//
//  SwiftDatasetAdapter.mm
//  NVFlareSDK
//
//  C++ adapter implementation to bridge Swift datasets to ETDataset interface
//

#import "SwiftDatasetAdapter.h"
#include <executorch/extension/tensor/tensor.h>
#include <algorithm>
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>

using namespace executorch::extension;

SwiftDatasetAdapter::SwiftDatasetAdapter(void* swiftObject) 
    : swiftObjectPtr(swiftObject), isDestroyed(false) {
    NSLog(@"SwiftDatasetAdapter constructor: created at %p with Swift object %p", this, swiftObject);
}

SwiftDatasetAdapter::~SwiftDatasetAdapter() {
    NSLog(@"SwiftDatasetAdapter destructor called for %p", this);
    
    if (isDestroyed) {
        NSLog(@"SwiftDatasetAdapter: Already destroyed! Double deletion detected!");
        return;
    }
    
    NSLog(@"SwiftDatasetAdapter: Setting isDestroyed flag");
    isDestroyed = true;
    
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
            swiftObjectPtr = nullptr;
        }
        
        NSLog(@"SwiftDatasetAdapter: Swift object pointer cleared");
    } else {
        NSLog(@"SwiftDatasetAdapter: No retained Swift object to release");
    }
    
    NSLog(@"SwiftDatasetAdapter destructor completed successfully for %p", this);
}

std::optional<SwiftDatasetAdapter::BatchType> SwiftDatasetAdapter::getBatch(size_t batchSize) {
    NSLog(@"SwiftDatasetAdapter::getBatch() called with size %zu", batchSize);
    
    if (isDestroyed || !swiftObjectPtr) {
        NSLog(@"SwiftDatasetAdapter::getBatch() object is destroyed or null");
        return std::nullopt;
    }
    
    @autoreleasepool {
        id swiftDataset = (__bridge id)swiftObjectPtr;
        if (!swiftDataset) {
            NSLog(@"SwiftDatasetAdapter::getBatch() bridged object is nil");
            return std::nullopt;
        }
        
        SEL getBatchSelector = @selector(getBatchWithSize:);
        if (![swiftDataset respondsToSelector:getBatchSelector]) {
            NSLog(@"SwiftDatasetAdapter::getBatch() object does not respond to getBatchWithSize:");
            return std::nullopt;
        }
        
        // Safer casting for objc_msgSend
        typedef NSArray* (*GetBatchFunc)(id, SEL, NSInteger);
        GetBatchFunc getBatchFunc = (GetBatchFunc)objc_msgSend;
        
        NSArray *result = getBatchFunc(swiftDataset, getBatchSelector, (NSInteger)batchSize);
        if (!result || result.count != 2) {
            NSLog(@"SwiftDatasetAdapter::getBatch() invalid result from Swift");
            return std::nullopt;
        }
        
        NSArray<NSNumber *> *inputs = result[0];
        NSArray<NSNumber *> *labels = result[1];
        
        if (!inputs || !labels) {
            NSLog(@"SwiftDatasetAdapter::getBatch() null inputs or labels");
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
        
        // Guard against division by zero
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
    
    if (isDestroyed || !swiftObjectPtr) {
        NSLog(@"SwiftDatasetAdapter::reset() object is destroyed or null");
        return;
    }
    
    @autoreleasepool {
        id swiftDataset = (__bridge id)swiftObjectPtr;
        if (!swiftDataset) {
            NSLog(@"SwiftDatasetAdapter::reset() bridged object is nil");
            return;
        }
        
        SEL resetSelector = @selector(reset);
        if ([swiftDataset respondsToSelector:resetSelector]) {
            typedef void (*ResetFunc)(id, SEL);
            ResetFunc resetFunc = (ResetFunc)objc_msgSend;
            resetFunc(swiftDataset, resetSelector);
            NSLog(@"SwiftDatasetAdapter::reset() completed");
        } else {
            NSLog(@"SwiftDatasetAdapter::reset() object does not respond to reset");
        }
    }
}

void SwiftDatasetAdapter::setShuffle(bool shuffle) {
    NSLog(@"SwiftDatasetAdapter::setShuffle() called with %s", shuffle ? "true" : "false");
    
    if (isDestroyed || !swiftObjectPtr) {
        NSLog(@"SwiftDatasetAdapter::setShuffle() object is destroyed or null");
        return;
    }
    
    @autoreleasepool {
        id swiftDataset = (__bridge id)swiftObjectPtr;
        if (!swiftDataset) {
            NSLog(@"SwiftDatasetAdapter::setShuffle() bridged object is nil");
            return;
        }
        
        SEL setShuffleSelector = @selector(setShuffle:);
        if ([swiftDataset respondsToSelector:setShuffleSelector]) {
            typedef void (*SetShuffleFunc)(id, SEL, BOOL);
            SetShuffleFunc setShuffleFunc = (SetShuffleFunc)objc_msgSend;
            setShuffleFunc(swiftDataset, setShuffleSelector, shuffle ? YES : NO);
            NSLog(@"SwiftDatasetAdapter::setShuffle() completed");
        } else {
            NSLog(@"SwiftDatasetAdapter::setShuffle() object does not respond to setShuffle:");
        }
    }
}

size_t SwiftDatasetAdapter::size() const {
    NSLog(@"SwiftDatasetAdapter::size() called on %p", this);
    
    if (isDestroyed) {
        NSLog(@"SwiftDatasetAdapter::size() called on destroyed object!");
        return 0;
    }
    
    if (!swiftObjectPtr) {
        NSLog(@"SwiftDatasetAdapter::size() Swift object pointer is null!");
        return 0;
    }
    
    NSLog(@"SwiftDatasetAdapter::size() calling Swift object at %p", swiftObjectPtr);
    
    @autoreleasepool {
        id swiftDataset = (__bridge id)swiftObjectPtr;
        if (!swiftDataset) {
            NSLog(@"SwiftDatasetAdapter::size() bridged object is nil!");
            return 0;
        }
        
        SEL sizeSelector = @selector(size);
        if (![swiftDataset respondsToSelector:sizeSelector]) {
            NSLog(@"SwiftDatasetAdapter::size() object does not respond to size selector");
            return 0;
        }
        
        NSLog(@"SwiftDatasetAdapter::size() calling size method on valid object");
        
        // Safer casting for objc_msgSend
        typedef NSInteger (*SizeFunc)(id, SEL);
        SizeFunc sizeFunc = (SizeFunc)objc_msgSend;
        
        NSInteger result = sizeFunc(swiftDataset, sizeSelector);
        NSLog(@"SwiftDatasetAdapter::size() result: %ld", (long)result);
        return static_cast<size_t>(result);
    }
}

size_t SwiftDatasetAdapter::inputDim() const {
    NSLog(@"SwiftDatasetAdapter::inputDim() called");
    
    if (isDestroyed || !swiftObjectPtr) {
        NSLog(@"SwiftDatasetAdapter::inputDim() object is destroyed or null");
        return 0;
    }
    
    @autoreleasepool {
        id swiftDataset = (__bridge id)swiftObjectPtr;
        if (!swiftDataset) {
            NSLog(@"SwiftDatasetAdapter::inputDim() bridged object is nil");
            return 0;
        }
        
        SEL inputDimSelector = @selector(inputDim);
        if ([swiftDataset respondsToSelector:inputDimSelector]) {
            typedef NSInteger (*InputDimFunc)(id, SEL);
            InputDimFunc inputDimFunc = (InputDimFunc)objc_msgSend;
            NSInteger result = inputDimFunc(swiftDataset, inputDimSelector);
            NSLog(@"SwiftDatasetAdapter::inputDim() result: %ld", (long)result);
            return static_cast<size_t>(result);
        } else {
            NSLog(@"SwiftDatasetAdapter::inputDim() object does not respond to inputDim");
            return 0;
        }
    }
}

size_t SwiftDatasetAdapter::labelDim() const {
    NSLog(@"SwiftDatasetAdapter::labelDim() called");
    
    if (isDestroyed || !swiftObjectPtr) {
        NSLog(@"SwiftDatasetAdapter::labelDim() object is destroyed or null");
        return 1; // Default to single label
    }
    
    @autoreleasepool {
        id swiftDataset = (__bridge id)swiftObjectPtr;
        if (!swiftDataset) {
            NSLog(@"SwiftDatasetAdapter::labelDim() bridged object is nil");
            return 1;
        }
        
        SEL labelDimSelector = @selector(labelDim);
        if ([swiftDataset respondsToSelector:labelDimSelector]) {
            typedef NSInteger (*LabelDimFunc)(id, SEL);
            LabelDimFunc labelDimFunc = (LabelDimFunc)objc_msgSend;
            NSInteger result = labelDimFunc(swiftDataset, labelDimSelector);
            NSLog(@"SwiftDatasetAdapter::labelDim() result: %ld", (long)result);
            return static_cast<size_t>(result);
        } else {
            NSLog(@"SwiftDatasetAdapter::labelDim() object does not respond to labelDim");
            return 1;
        }
    }
} 