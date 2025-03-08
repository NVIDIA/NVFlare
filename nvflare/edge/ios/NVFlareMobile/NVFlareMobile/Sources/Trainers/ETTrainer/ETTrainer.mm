//
//  ETTrainer.mm
//  NVFlareMobile
//
//

#import "ETTrainer.h"
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
#include "ETDataset.hpp"
#include "Constants.h"


using namespace ::executorch::extension;


// Helper functions for printing tensors
void printElements(const torch::executor::Tensor& tensor,
                  torch::executor::ArrayRef<int> sizes,
                  torch::executor::ArrayRef<int> strides,
                  const float* data_ptr,
                  std::vector<int>& indices,
                  int dim) {
    if (dim == sizes.size()) {
        // Base case: we have a full index, calculate the linear index
        int64_t linear_index = 0;
        for (int i = 0; i < sizes.size(); ++i) {
            linear_index += indices[i] * strides[i];
        }
        // Access the element in the raw data
        float value = data_ptr[linear_index];
        // Print the element
        NSString* indexStr = @"";
        for (int i = 0; i < indices.size(); ++i) {
            indexStr = [indexStr stringByAppendingFormat:@"%d%@", indices[i],
                       (i < indices.size() - 1) ? @", " : @""];
        }
        NSLog(@"arr[%@] = %f", indexStr, value);
        return;
    }
    
    // Recursive case: loop through the current dimension
    for (int64_t i = 0; i < sizes[dim]; ++i) {
        indices.push_back(i);
        printElements(tensor, sizes, strides, data_ptr, indices, dim + 1);
        indices.pop_back();
    }
}

void printTensorElements(const torch::executor::Tensor& tensor) {
    auto strides = tensor.strides();
    auto data_ptr = tensor.const_data_ptr<float>();
    auto sizes = tensor.sizes();
    
    std::vector<int> indices;
    printElements(tensor, sizes, strides, data_ptr, indices, 0);
}

void printMap(const std::map<executorch::aten::string_view, executorch::aten::Tensor>& map) {
    for (const auto& pair : map) {
        NSLog(@"Key: %s", pair.first.data());
        printTensorElements(pair.second);
    }
}



@implementation ETTrainer {
    std::unique_ptr<training::TrainingModule> _training_module;
    NSDictionary<NSString *, id> *_meta;
    std::unique_ptr<ETDataset> _dataset;
}

- (instancetype)initWithModelBase64:(NSString *)modelBase64
                             meta:(NSDictionary<NSString *, id> *)meta {
    self = [super init];
    if (self) {
        _meta = meta;
        
        // Initialize dataset based on meta configuration
        NSString *datasetType = _meta[kMetaKeyDatasetType];
        
        if ([datasetType isEqualToString:kDatasetTypeCIFAR10]) {
            NSDataAsset *dataAsset = [[NSDataAsset alloc] initWithName:@"data_batch_1"];
            if (!dataAsset) {
                NSLog(@"Failed to load CIFAR-10 data from assets");
                return nil;
            }
            NSData *binaryData = dataAsset.data;
            const char *bytes = (const char *)[binaryData bytes];
            NSUInteger length = [binaryData length];
            std::istringstream dataStream(std::string(bytes, length));
            _dataset = std::make_unique<CIFAR10Dataset>(dataStream);
            
        } else if ([datasetType isEqualToString:kDatasetTypeXOR]) {
            _dataset = std::make_unique<XORDataset>();
            
        } else {
            NSLog(@"Unknown dataset type: %@", datasetType);
            return nil;
        }
        
        // Decode base64 string to temporary file
        NSData *modelData = [[NSData alloc] initWithBase64EncodedString:modelBase64
                                                              options:0];
        if (!modelData) {
            NSLog(@"Failed to decode base64 model data");
            return nil;
        }
        
        // Write to temporary file
        NSString *tempPath = [NSTemporaryDirectory() stringByAppendingPathComponent:@"temp_model.pte"];
        if (![modelData writeToFile:tempPath atomically:YES]) {
            NSLog(@"Failed to write model data to temporary file");
            return nil;
        }
        
        @try {
            // Load model using FileDataLoader
            auto model_result = FileDataLoader::from(tempPath.UTF8String);
            if (!model_result.ok()) {
                NSLog(@"Failed to load model file");
                return nil;
            }
            
            auto loader = std::make_unique<FileDataLoader>(std::move(model_result.get()));
            _training_module = std::make_unique<training::TrainingModule>(std::move(loader));
            
            // Clean up temporary file
            [[NSFileManager defaultManager] removeItemAtPath:tempPath error:nil];
            
        } @catch (NSException *exception) {
            NSLog(@"Failed to initialize training module: %@", exception);
            return nil;
        }
    }
    return self;
}

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

+ (void)printTensorDictionary:(NSDictionary<NSString *, id> *)dict {
    NSLog(@"Dictionary Contents ===============");
    for (NSString *key in dict) {
        NSLog(@"Tensor: %@", key);
        NSDictionary *tensorInfo = dict[key];
        
        NSArray *sizes = tensorInfo[@"sizes"];
        NSLog(@"  Sizes: %@", sizes);
        
        NSArray *strides = tensorInfo[@"strides"];
        NSLog(@"  Strides: %@", strides);
        
        NSArray *data = tensorInfo[@"data"];
        NSLog(@"  Data[%lu]: [", (unsigned long)data.count);
        // Print first few and last few elements
        for (int i = 0; i < data.count; i++) {
            NSLog(@"    [%d]: %@", i, data[i]);
        }
        NSLog(@"  ]");
    }
    NSLog(@"End Dictionary Contents ===========");
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
    if (!_training_module) {
        NSLog(@"Training module not initialized");
        return @{};
    }

    @try {
        int batchSize = [_meta[kMetaKeyBatchSize] intValue];
        auto data_set = _dataset->getBatch(batchSize);
        
        // Get initial parameters
        auto param_res = _training_module->named_parameters("forward");
        if (param_res.error() != executorch::runtime::Error::Ok) {
            NSLog(@"Failed to get named parameters");
            return @{};
        }
        
        auto initial_params = param_res.get();
        NSDictionary<NSString *, id>* old_params = [ETTrainer toTensorDictionary:initial_params];
    
//        NSLog(@"Initial Params Start ==============");
//        [ETTrainer printTensorDictionary:old_params];
//        NSLog(@"Initial Params End ================");
        
        // Configure optimizer
        float learningRate = [_meta[kMetaKeyLearningRate] floatValue];
        training::optimizer::SGDOptions options{learningRate};
        training::optimizer::SGD optimizer(param_res.get(), options);
        
        // Train the model
        NSInteger totalEpochs = [_meta[kMetaKeyTotalEpochs] integerValue];
        for (int i = 0; i < totalEpochs; i++) {
            size_t index = i % data_set.size();
            auto& data = data_set[index];
            
            const auto& results = _training_module->execute_forward_backward(
                "forward",
                {*data.first, *data.second}
            );
            
            if (results.error() != executorch::runtime::Error::Ok) {
                NSLog(@"Failed to execute forward_backward");
                return @{};
            }
            
            if (i % 500 == 0 || i == totalEpochs - 1) {
                NSLog(@"Step %d, Loss %f, Input [%.0f, %.0f], Prediction %lld, Label %lld",
                    i,
                    results.get()[0].toTensor().const_data_ptr<float>()[0],
                    data.first->const_data_ptr<float>()[0],
                    data.first->const_data_ptr<float>()[1],
                    results.get()[1].toTensor().const_data_ptr<int64_t>()[0],
                    data.second->const_data_ptr<int64_t>()[0]);
            }
            
            optimizer.step(_training_module->named_gradients("forward").get());
        }
        
//        NSLog(@"Grad Start ==============");
//        printMap(_training_module->named_gradients("forward").get());
//        NSLog(@"Grad End ================");
//
//        NSLog(@"Old Params Start ==============");
//        [ETTrainer printTensorDictionary:old_params];
//        NSLog(@"Old Params End ================");


        NSDictionary<NSString *, id>* final_params = [ETTrainer toTensorDictionary:param_res.get()];
        
//        NSLog(@"New Params Start ==============");
//        [ETTrainer printTensorDictionary:final_params];
//        NSLog(@"New Params End ================");
        
        auto tensor_diff = [ETTrainer calculateTensorDifference:old_params newDict:final_params];
        
//        NSLog(@"Diff Start ==============");
//        [ETTrainer printTensorDictionary:tensor_diff];
//        NSLog(@"Diff End ================");

        return tensor_diff;
        
    } @catch (NSException *exception) {
        NSLog(@"Training failed: %@", exception);
        return @{};
    }
}

@end
