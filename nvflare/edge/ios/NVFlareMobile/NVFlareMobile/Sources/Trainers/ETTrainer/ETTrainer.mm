//
//  ETTrainer.mm
//  NVFlareMobile
//
//

#import "ETTrainer.h"

#import <executorch/extension/module/module.h>
#import <executorch/extension/tensor/tensor.h>
#import <executorch/extension/data_loader/file_data_loader.h>
#import <executorch/extension/training/module/training_module.h>
#import <executorch/extension/training/optimizer/sgd.h>
#include <map>
#include <string>
#include <vector>


// Comments:
//
// (1) how to serialize/deserialize for server side weights??
// (2) how to serialize/deserialize for client side weights??
// (3) how to get/load weights from a ".pte" file?? or need to exchange the whole ".pte" file??
// (4) then how to HTTP request to get/send weights to the WebAPI (iOS question)
// (5) short-term goal: do a demo in GTC
// (6) figure out the long-term goal (investigate LiteRT, Onnx runtime):
//     - one piece of code for all platforms: Android, iOS, browser, edge/wearables?
//     - support training/inference
//     - support deep learning and XGBoost
//     - is active not abandoned


// (1) How to load weights into client/device => use PTE
// (2) Does loading weights in the server will take effect when exporting => yes
// (3) from client to server use JSON




// *** the ".pte" is in base64 string for now NOT binary
// *** send the model difference instead of gradient to support multiple local epochs




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
}

- (instancetype)initWithModelBase64:(NSString *)modelBase64
                             meta:(NSDictionary<NSString *, id> *)meta {
    self = [super init];
    if (self) {
        _meta = meta;
        
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

- (NSDictionary<NSString *, id> *)toNSDictionary:(const std::map<executorch::aten::string_view, executorch::aten::Tensor>&)map {
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

- (NSDictionary<NSString *, id> *)train {
    if (!_training_module) {
        NSLog(@"Training module not initialized");
        return @{};
    }
    
    @try {
        // Create full data set of input and labels (XOR example)
        std::vector<std::pair<TensorPtr,TensorPtr>> data_set;
        data_set.push_back( // XOR(1, 1) = 0
          {make_tensor_ptr<float>({1, 2}, {1, 1}),
           make_tensor_ptr<int64_t>({1}, {0})});
        data_set.push_back( // XOR(0, 0) = 0
          {make_tensor_ptr<float>({1, 2}, {0, 0}),
           make_tensor_ptr<int64_t>({1}, {0})});
        data_set.push_back( // XOR(1, 0) = 1
          {make_tensor_ptr<float>({1, 2}, {1, 0}),
           make_tensor_ptr<int64_t>({1}, {1})});
        data_set.push_back( // XOR(0, 1) = 1
          {make_tensor_ptr<float>({1, 2}, {0, 1}),
           make_tensor_ptr<int64_t>({1}, {1})});
        
        // Get initial parameters
        auto param_res = _training_module->named_parameters("forward");
        if (param_res.error() != executorch::runtime::Error::Ok) {
            NSLog(@"Failed to get named parameters");
            return @{};
        }
        
        NSLog(@"Initial Params Start ==============");
        printMap(param_res.get());
        NSLog(@"Initial Params End ================");
        
        // Configure optimizer
        float learningRate = [_meta[@"learning_rate"] floatValue];
        training::optimizer::SGDOptions options{learningRate};
        training::optimizer::SGD optimizer(param_res.get(), options);
        
        // Train the model
        NSInteger totalEpochs = [_meta[@"total_epochs"] integerValue];
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
        
        NSLog(@"Grad Start ==============");
        printMap(_training_module->named_gradients("forward").get());
        NSLog(@"Grad End ================");
        
        NSLog(@"New Params Start ==============");
        printMap(_training_module->named_gradients("forward").get());
        NSLog(@"New Params End ================");
        
        // Use the instance method directly
        return [self toNSDictionary:_training_module->named_gradients("forward").get()];
        
    } @catch (NSException *exception) {
        NSLog(@"Training failed: %@", exception);
        return @{};
    }
}

@end
