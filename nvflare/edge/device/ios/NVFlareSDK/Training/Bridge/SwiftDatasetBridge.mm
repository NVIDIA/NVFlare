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

+ (void*)createDatasetAdapter:(id)nvflareDataset {
    NSLog(@"SwiftDatasetBridge: Creating adapter for NVFlareDataset: %@", nvflareDataset);
    
    // Validate the NVFlareDataset first
    if (!nvflareDataset) {
        NSLog(@"SwiftDatasetBridge: NVFlareDataset is nil!");
        return nullptr;
    }
    
    // Test if the object responds to required methods
    SEL sizeSelector = sel_registerName("size");
    if (![nvflareDataset respondsToSelector:sizeSelector]) {
        NSLog(@"SwiftDatasetBridge: NVFlareDataset does not respond to size selector!");
        return nullptr;
    }
    
    SEL getNextBatchSelector = sel_registerName("getNextBatchWithBatchSize:");
    if (![nvflareDataset respondsToSelector:getNextBatchSelector]) {
        NSLog(@"SwiftDatasetBridge: NVFlareDataset does not respond to getNextBatchWithBatchSize selector!");
        return nullptr;
    }
    
    // Test the size method before creating the adapter
    @try {
        NSInteger testSize = ((NSInteger (*)(id, SEL))objc_msgSend)(nvflareDataset, sizeSelector);
        NSLog(@"SwiftDatasetBridge: Test size call successful: %ld", (long)testSize);
    } @catch (NSException *exception) {
        NSLog(@"SwiftDatasetBridge: Test size call failed: %@", exception);
        return nullptr;
    }
    
    // Standard ARC approach - store strong reference
    id retainedDataset = nvflareDataset;
    
    // Create the C++ adapter with proper retained reference
    void* retainedPtr = (void*)CFBridgingRetain(retainedDataset);
    NSLog(@"SwiftDatasetBridge: CFBridgingRetain returned: %p", retainedPtr);
    
    if (!retainedPtr) {
        NSLog(@"SwiftDatasetBridge: CFBridgingRetain returned null!");
        return nullptr;
    }
    
    SwiftDatasetAdapter* adapter = new SwiftDatasetAdapter(retainedPtr);
    
    NSLog(@"SwiftDatasetBridge: Created C++ dataset adapter at %p for NVFlareDataset", adapter);
    return static_cast<void*>(adapter);
}

+ (void)destroyDatasetAdapter:(void*)dataset {
    NSLog(@"SwiftDatasetBridge: destroyDatasetAdapter called with pointer: %p", dataset);
    
    if (!dataset) {
        NSLog(@"SwiftDatasetBridge: dataset pointer is null, nothing to destroy");
        return;
    }
    
    @try {
        SwiftDatasetAdapter* adapter = static_cast<SwiftDatasetAdapter*>(dataset);
        
        NSLog(@"SwiftDatasetBridge: About to delete adapter at %p", adapter);
        
        delete adapter;
        
        NSLog(@"SwiftDatasetBridge: Successfully destroyed C++ dataset adapter");
    } @catch (NSException *exception) {
        NSLog(@"SwiftDatasetBridge: Exception during adapter destruction: %@", exception);
    }
}

@end 