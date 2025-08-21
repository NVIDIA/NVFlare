//
//  NVFlareConstants.h
//  NVFlare iOS SDK
//
//  Objective-C constants that mirror Swift constants for C++ code access
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// Protocol Meta Keys - These should match the values in NVFlareProtocolConstants.swift
extern NSString * const kNVFlareMetaKeyBatchSize;
extern NSString * const kNVFlareMetaKeyLearningRate;
extern NSString * const kNVFlareMetaKeyTotalEpochs;
extern NSString * const kNVFlareMetaKeyDatasetShuffle;
extern NSString * const kNVFlareMetaKeyDatasetType;

// Training Methods - These should match the values in TrainingConstants.swift
extern NSString * const kNVFlareMethodCNN;
extern NSString * const kNVFlareMethodMLP;

// Dataset Types - These should match the values in TrainingConstants.swift
extern NSString * const kNVFlareDatasetTypeCIFAR10;
extern NSString * const kNVFlareDatasetTypeXOR;

NS_ASSUME_NONNULL_END
