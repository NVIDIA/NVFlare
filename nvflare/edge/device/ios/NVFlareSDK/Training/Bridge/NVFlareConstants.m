//
//  NVFlareConstants.m
//  NVFlare iOS SDK
//
//  Objective-C constants implementation - values must match Swift constants
//

#import "NVFlareConstants.h"

// Protocol Meta Keys - These values MUST match NVFlareProtocolConstants.swift
NSString * const kNVFlareMetaKeyBatchSize = @"batch_size";
NSString * const kNVFlareMetaKeyLearningRate = @"learning_rate";
NSString * const kNVFlareMetaKeyTotalEpochs = @"total_epochs";
NSString * const kNVFlareMetaKeyDatasetShuffle = @"dataset_shuffle";
NSString * const kNVFlareMetaKeyDatasetType = @"dataset_type";

// Training Methods - These values MUST match TrainingConstants.swift
NSString * const kNVFlareMethodCNN = @"cnn";
NSString * const kNVFlareMethodMLP = @"mlp";

// Dataset Types - These values MUST match TrainingConstants.swift
NSString * const kNVFlareDatasetTypeCIFAR10 = @"cifar10";
NSString * const kNVFlareDatasetTypeXOR = @"xor";
