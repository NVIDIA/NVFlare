//
//  SwiftDatasetBridge.h
//  NVFlareSDK
//
//  Bridge to create C++ dataset adapters from Swift datasets
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Bridge to create C++ dataset adapters from NVFlare datasets
@interface SwiftDatasetBridge : NSObject

/// Create a C++ dataset adapter from an NVFlareDataset
/// @param nvflareDataset The NVFlareDataset object
/// @return Pointer to C++ ETDataset* that can be used with ETTrainer
+ (void*)createDatasetAdapter:(id)nvflareDataset;

/// Destroy a C++ dataset adapter
/// @param dataset Pointer to C++ ETDataset* created by createDatasetAdapter
+ (void)destroyDatasetAdapter:(void*)dataset;

@end

NS_ASSUME_NONNULL_END 