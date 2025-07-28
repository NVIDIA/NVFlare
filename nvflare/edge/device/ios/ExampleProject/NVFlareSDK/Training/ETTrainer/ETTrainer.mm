//
//  ETTrainer.mm
//  NVFlareMobile
//
//

#include "ETTrainer.h"
#import <UIKit/UIKit.h>
#import <executorch/extension/module/module.h>
#import <executorch/extension/tensor/tensor.h>
#import <executorch/extension/data_loader/file_data_loader.h>
#import <executorch/extension/training/module/training_module.h>
#import <executorch/extension/training/optimizer/sgd.h>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include "ETDataset.h"
#include "Constants.h"
#include "ETDebugUtils.h"

using namespace ::executorch::extension;

@implementation ETTrainer {
    std::unique_ptr<training::TrainingModule> _training_module;
    NSDictionary<NSString *, id> *_meta;
    std::unique_ptr<ETDataset> _dataset;
}

// loadDataset method removed - dataset passed directly to initializer

- (std::unique_ptr<training::TrainingModule>)loadModel:(NSString *)modelBase64 {
    // Decode base64 string to temporary file
    NSData *modelData = [[NSData alloc] initWithBase64EncodedString:modelBase64 options:0];
    if (!modelData) {
        return nullptr;
    }
    
    // Write to temporary file
    NSString *tempPath = [NSTemporaryDirectory() stringByAppendingPathComponent:@"temp_model.pte"];
    if (![modelData writeToFile:tempPath atomically:YES]) {
        return nullptr;
    }
    
    std::unique_ptr<training::TrainingModule> module;
    @try {
        // Load model using FileDataLoader
        auto model_result = FileDataLoader::from(tempPath.UTF8String);
        if (!model_result.ok()) {
            return nullptr;
        }
        
        auto loader = std::make_unique<FileDataLoader>(std::move(model_result.get()));
        module = std::make_unique<training::TrainingModule>(std::move(loader));
        
    } @catch (NSException *exception) {
        module = nullptr;
    }
    
    // Clean up temporary file
    [[NSFileManager defaultManager] removeItemAtPath:tempPath error:nil];
    
    return module;
}

/// Primary initializer - accepts C++ dataset directly
- (instancetype)initWithModelBase64:(NSString *)modelBase64
                             meta:(NSDictionary<NSString *, id> *)meta
                          dataset:(void *)cppDataset {
    NSLog(@"🚀 ETTrainer: Initialization started with app's C++ dataset");
    self = [super init];
    if (self) {
        _meta = meta;
        
        // Use app's C++ dataset directly
        if (!cppDataset) {
            NSLog(@"❌ ETTrainer: App provided null C++ dataset");
            return nil;
        }
        
        NSLog(@"🔍 ETTrainer: cppDataset pointer = %p", cppDataset);
        
        // Cast to our expected C++ dataset type
        ETDataset* dataset = static_cast<ETDataset*>(cppDataset);
        
        NSLog(@"🔍 ETTrainer: After cast, dataset pointer = %p", dataset);
        
        // Create unique_ptr
        _dataset = std::unique_ptr<ETDataset>(dataset);
        
        NSLog(@"🔍 ETTrainer: unique_ptr created, about to test object validity...");
        
        // Test if the pointer is valid before calling methods
        if (!_dataset.get()) {
            NSLog(@"❌ ETTrainer: unique_ptr is null after creation");
            return nil;
        }
        
        NSLog(@"🔍 ETTrainer: unique_ptr.get() = %p", _dataset.get());
        
        // Try a safer approach - use raw pointer first
        ETDataset* rawPtr = _dataset.get();
        NSLog(@"🔍 ETTrainer: Raw pointer = %p", rawPtr);
        
        // Safely try to access the dataset
        @try {
            NSLog(@"🔍 ETTrainer: About to call size() on raw pointer...");
            size_t datasetSize = rawPtr->size();
            NSLog(@"✅ ETTrainer: App's C++ dataset ready (size: %zu)", datasetSize);
        } @catch (NSException *exception) {
            NSLog(@"❌ ETTrainer: Exception accessing dataset: %@", exception);
            return nil;
        } @catch (...) {
            NSLog(@"❌ ETTrainer: C++ exception accessing dataset");
            return nil;
        }
        
        // Load model
        NSLog(@"🤖 ETTrainer: Loading ExecutorTorch model");
        _training_module = [self loadModel:modelBase64];
        if (!_training_module) {
            NSLog(@"❌ ETTrainer: Failed to load ExecutorTorch model");
            return nil;
        }
        NSLog(@"✅ ETTrainer: ExecutorTorch model loaded successfully");
    }
    NSLog(@"🎯 ETTrainer: Initialization complete with app's C++ dataset");
    return self;
}

// Helper methods removed - app provides C++ dataset directly

+ (NSDictionary<NSString *, id> *)toTensorDictionary:(const std::map<executorch::aten::string_view, executorch::aten::Tensor>&)map {
    NSMutableDictionary *tensorDict = [NSMutableDictionary dictionary];
    
    for (const auto& pair : map) {
        NSString *key = [NSString stringWithUTF8String:pair.first.data()];
        executorch::aten::Tensor tensor = pair.second;

        NSMutableDictionary *singleTensorDict = [NSMutableDictionary dictionary];
        
        auto strides = tensor.strides();
        auto sizes = tensor.sizes();
        auto data_ptr = tensor.const_data_ptr<float>();
        
        NSMutableArray *stridesArray = [NSMutableArray arrayWithCapacity:strides.size()];
        for (size_t i = 0; i < strides.size(); ++i) {
            [stridesArray addObject:@(strides[i])];
        }
        singleTensorDict[@"strides"] = stridesArray;
        
        NSMutableArray *sizesArray = [NSMutableArray arrayWithCapacity:sizes.size()];
        for (size_t i = 0; i < sizes.size(); ++i) {
            [sizesArray addObject:@(sizes[i])];
        }
        singleTensorDict[@"sizes"] = sizesArray;

        NSMutableArray *dataArray = [NSMutableArray arrayWithCapacity:tensor.numel()];
        for (size_t i = 0; i < tensor.numel(); ++i) {
            [dataArray addObject:@(data_ptr[i])];
        }
        singleTensorDict[@"data"] = dataArray;
        
        tensorDict[key] = singleTensorDict;
    }

    return tensorDict;
}

+ (NSDictionary<NSString *, id> *)calculateTensorDifference:(NSDictionary<NSString *, id> *)oldDict
                                                   newDict:(NSDictionary<NSString *, id> *)newDict {
    NSMutableDictionary *diffDict = [NSMutableDictionary dictionary];
    
    for (NSString *key in oldDict) {
        NSDictionary *oldTensor = oldDict[key];
        NSDictionary *newTensor = newDict[key];
        
        if (!newTensor) {
            NSLog(@"Warning: Tensor %@ not found in new parameters", key);
            continue;
        }
        
        NSArray *oldData = oldTensor[@"data"];
        NSArray *newData = newTensor[@"data"];
        
        if (oldData.count != newData.count) {
            NSLog(@"Warning: Tensor %@ size mismatch: old=%lu new=%lu",
                  key, (unsigned long)oldData.count, (unsigned long)newData.count);
            continue;
        }
        
        NSMutableArray *diffData = [NSMutableArray arrayWithCapacity:oldData.count];
        for (NSUInteger i = 0; i < oldData.count; i++) {
            float oldVal = [oldData[i] floatValue];
            float newVal = [newData[i] floatValue];
            float diff = newVal - oldVal;
            [diffData addObject:@(diff)];
        }
        
        NSMutableDictionary *diffTensor = [NSMutableDictionary dictionary];
        diffTensor[@"sizes"] = oldTensor[@"sizes"];     // Keep original sizes
        diffTensor[@"strides"] = oldTensor[@"strides"]; // Keep original strides
        diffTensor[@"data"] = diffData;                 // Store differences
        
        diffDict[key] = diffTensor;
    }
    
    return diffDict;
}


- (NSDictionary<NSString *, id> *)train {
    NSLog(@"ETTrainer: Starting train()");
    if (!_training_module) {
        NSLog(@"ETTrainer: Training module not initialized");
        return @{};
    }

    @try {
        int batchSize = [_meta[kMetaKeyBatchSize] intValue];
        NSLog(@"ETTrainer: Using batch size: %d", batchSize);
        
        // Get initial parameters
        auto param_res = _training_module->named_parameters("forward");
        if (param_res.error() != executorch::runtime::Error::Ok) {
            NSLog(@"ETTrainer: Failed to get named parameters");
            return @{};
        }
        
        auto initial_params = param_res.get();
        NSDictionary<NSString *, id>* old_params = [ETTrainer toTensorDictionary:initial_params];
        NSLog(@"ETTrainer: Got initial parameters");
        
        printTensorDictionary(old_params, @"Initial Params");
        

        // Configure optimizer
        float learningRate = [_meta[kMetaKeyLearningRate] floatValue];
        training::optimizer::SGDOptions options{learningRate};
        training::optimizer::SGD optimizer(param_res.get(), options);
        
        // Train the model
        NSInteger totalEpochs = [_meta[kMetaKeyTotalEpochs] integerValue];
        int totalSteps = 0;
        size_t datasetSize = _dataset->size();
        size_t numBatchesPerEpoch = (datasetSize + batchSize - 1) / batchSize;  // Ceiling division
        
        for (int epoch = 0; epoch < totalEpochs; epoch++) {
            _dataset->reset(); // Reset dataset at the start of each epoch
            
            for (size_t batchIdx = 0; batchIdx < numBatchesPerEpoch; batchIdx++) {
                auto batchOpt = _dataset->getBatch(batchSize);
                if (!batchOpt) break;  // End of dataset
                
                const auto& [input, label] = *batchOpt;
                const auto& results = _training_module->execute_forward_backward(
                    "forward",
                    {*input, *label}
                );
                
                if (results.error() != executorch::runtime::Error::Ok) {
                    NSLog(@"Failed to execute forward_backward");
                    return @{};
                }
                
                size_t samplesProcessed = batchIdx * batchSize + input->sizes()[0];
                if (totalSteps % 500 == 0 || (epoch == totalEpochs - 1 && batchIdx == numBatchesPerEpoch - 1)) {
                    NSLog(@"Epoch %d/%lld, Progress %.1f%%, Step %d, Loss %f, Prediction %lld, Label %lld",
                        epoch + 1, (long long)totalEpochs,
                        (float)samplesProcessed * 100 / datasetSize,
                        totalSteps,
                        results.get()[0].toTensor().const_data_ptr<float>()[0],
                        results.get()[1].toTensor().const_data_ptr<int64_t>()[0],
                        label->const_data_ptr<int64_t>()[0]);
                }
                
                optimizer.step(_training_module->named_gradients("forward").get());
                totalSteps++;
            }
        }
        
        NSDictionary<NSString *, id>* final_params = [ETTrainer toTensorDictionary:param_res.get()];
        
        
        printTensorDictionary(final_params, @"Final Params");
       
        
        auto tensor_diff = [ETTrainer calculateTensorDifference:old_params newDict:final_params];
        
       
        printTensorDictionary(tensor_diff, @"Tensor Diff");

        return tensor_diff;
        
    } @catch (NSException *exception) {
        NSLog(@"Training failed: %@", exception);
        return @{};
    }
}

@end
