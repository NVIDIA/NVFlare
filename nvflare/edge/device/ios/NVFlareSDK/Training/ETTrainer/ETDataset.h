//
//  ETDataset.h
//  NVFlare iOS SDK
//
//  Generic dataset interface for ExecutorTorch integration
//

#pragma once

#include <vector>
#include <optional>
#include <executorch/extension/tensor/tensor.h>

// Generic Dataset template (base interface for app implementations)
template<typename InputType, typename LabelType>
class Dataset {
public:
    using BatchType = std::pair<InputType, LabelType>;
    
    virtual ~Dataset() = default;
    
    // Get a single batch of data, returns nullopt when all data is used
    virtual std::optional<BatchType> getBatch(size_t batchSize) = 0;
    
    // Reset the dataset iterator
    virtual void reset() = 0;
    
    // Get total number of samples in dataset
    virtual size_t size() const = 0;
    
    // Get dimensions of input and label data
    virtual size_t inputDim() const = 0;
    virtual size_t labelDim() const = 0;
    
    // Control shuffling behavior
    virtual void setShuffle(bool shuffle) = 0;
};

// ETTrainer specific dataset using TensorPtr (for app implementations)
using ETDataset = Dataset<executorch::extension::TensorPtr, executorch::extension::TensorPtr>;
