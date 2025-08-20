//
//  SwiftDatasetBridge.h
//  NVFlareSDK
//
//  Bridge to create C++ dataset adapters from Swift datasets
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Bridge to create C++ dataset adapters from Swift datasets
@interface SwiftDatasetBridge : NSObject

/// Create a C++ dataset adapter from a Swift dataset
/// @param swiftDataset The Swift dataset object
/// @return Pointer to C++ ETDataset* that can be used with ETTrainer
+ (void*)createDatasetAdapter:(id)swiftDataset;

/// Destroy a C++ dataset adapter
/// @param dataset Pointer to C++ ETDataset* created by createDatasetAdapter
+ (void)destroyDatasetAdapter:(void*)dataset;

@end

NS_ASSUME_NONNULL_END 