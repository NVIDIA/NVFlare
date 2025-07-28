//
//  SwiftDatasetBridge.mm
//  NVFlareSDK
//
//  Bridge implementation to create C++ dataset adapters from Swift datasets
//

#import "SwiftDatasetBridge.h"
#import "SwiftDatasetAdapter.h"
#include <functional>
#include <optional>
#include <vector>
#import <objc/runtime.h>
#import <objc/message.h>

@implementation SwiftDatasetBridge

+ (void*)createDatasetAdapter:(id)swiftDataset {
    // We'll use dynamic method calls since we can't directly access Swift protocols from Objective-C++
    // The Swift side will ensure the object implements the required methods
    
    // Create C++ callbacks that call Swift methods
    auto getBatchFunc = [swiftDataset](size_t batchSize) -> std::optional<std::pair<std::vector<float>, std::vector<int64_t>>> {
        @autoreleasepool {
            // Use dynamic method call for Swift method getBatch(size:)
            SEL selector = NSSelectorFromString(@"getBatchWithSize:");
            if (![swiftDataset respondsToSelector:selector]) {
                NSLog(@"SwiftDatasetBridge: Object does not respond to getBatchWithSize:");
                return std::nullopt;
            }
            
            NSArray *result = ((NSArray *(*)(id, SEL, NSInteger))objc_msgSend)(swiftDataset, selector, (NSInteger)batchSize);
            if (!result || result.count != 2) {
                return std::nullopt;
            }
            
            NSArray<NSNumber *> *inputs = result[0];
            NSArray<NSNumber *> *labels = result[1];
            
            if (!inputs || !labels) {
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
            
            return std::make_pair(std::move(inputVec), std::move(labelVec));
        }
    };
    
    auto resetFunc = [swiftDataset]() {
        @autoreleasepool {
            SEL selector = @selector(reset);
            if ([swiftDataset respondsToSelector:selector]) {
                ((void (*)(id, SEL))objc_msgSend)(swiftDataset, selector);
            }
        }
    };
    
    auto sizeFunc = [swiftDataset]() -> size_t {
        @autoreleasepool {
            SEL selector = @selector(size);
            if ([swiftDataset respondsToSelector:selector]) {
                NSInteger result = ((NSInteger (*)(id, SEL))objc_msgSend)(swiftDataset, selector);
                return static_cast<size_t>(result);
            }
            return 0;
        }
    };
    
    auto inputDimFunc = [swiftDataset]() -> size_t {
        @autoreleasepool {
            SEL selector = @selector(inputDim);
            if ([swiftDataset respondsToSelector:selector]) {
                NSInteger result = ((NSInteger (*)(id, SEL))objc_msgSend)(swiftDataset, selector);
                return static_cast<size_t>(result);
            }
            return 0;
        }
    };
    
    auto labelDimFunc = [swiftDataset]() -> size_t {
        @autoreleasepool {
            SEL selector = @selector(labelDim);
            if ([swiftDataset respondsToSelector:selector]) {
                NSInteger result = ((NSInteger (*)(id, SEL))objc_msgSend)(swiftDataset, selector);
                return static_cast<size_t>(result);
            }
            return 1; // Default to single label
        }
    };
    
    auto setShuffleFunc = [swiftDataset](bool shuffle) {
        @autoreleasepool {
            SEL selector = NSSelectorFromString(@"setShuffle:");
            if ([swiftDataset respondsToSelector:selector]) {
                ((void (*)(id, SEL, BOOL))objc_msgSend)(swiftDataset, selector, shuffle ? YES : NO);
            }
        }
    };
    
    // Create the C++ adapter
    SwiftDatasetAdapter* adapter = new SwiftDatasetAdapter(
        getBatchFunc, resetFunc, sizeFunc, inputDimFunc, labelDimFunc, setShuffleFunc
    );
    
    NSLog(@"SwiftDatasetBridge: Created C++ dataset adapter for Swift dataset");
    return static_cast<void*>(adapter);
}

+ (void)destroyDatasetAdapter:(void*)dataset {
    if (dataset) {
        SwiftDatasetAdapter* adapter = static_cast<SwiftDatasetAdapter*>(dataset);
        delete adapter;
        NSLog(@"SwiftDatasetBridge: Destroyed C++ dataset adapter");
    }
}

@end 