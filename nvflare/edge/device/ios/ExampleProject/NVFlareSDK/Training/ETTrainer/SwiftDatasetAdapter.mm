//
//  SwiftDatasetAdapter.mm
//  NVFlareSDK
//
//  C++ adapter implementation to bridge Swift datasets to ETDataset interface
//

#import "SwiftDatasetAdapter.h"
#include <executorch/extension/tensor/tensor.h>
#include <algorithm>

using namespace executorch::extension;

SwiftDatasetAdapter::SwiftDatasetAdapter(
    std::function<std::optional<std::pair<std::vector<float>, std::vector<int64_t>>>(size_t)> getBatch,
    std::function<void()> reset,
    std::function<size_t()> size,
    std::function<size_t()> inputDim,
    std::function<size_t()> labelDim,
    std::function<void(bool)> setShuffle
) : getBatchFunc(getBatch), resetFunc(reset), sizeFunc(size), 
    inputDimFunc(inputDim), labelDimFunc(labelDim), setShuffleFunc(setShuffle) {
}

std::optional<SwiftDatasetAdapter::BatchType> SwiftDatasetAdapter::getBatch(size_t batchSize) {
    if (!getBatchFunc) {
        return std::nullopt;
    }
    
    auto result = getBatchFunc(batchSize);
    if (!result) {
        return std::nullopt;
    }
    
    auto [inputs, labels] = *result;
    if (inputs.empty() || labels.empty()) {
        return std::nullopt;
    }
    
    // Calculate input dimensions
    size_t batchSizeActual = labels.size();
    size_t inputDim = inputs.size() / batchSizeActual;
    
    // Create tensors
    auto inputTensor = make_tensor_ptr<float>(
        {static_cast<int>(batchSizeActual), static_cast<int>(inputDim)}, 
        std::move(inputs)
    );
    
    auto labelTensor = make_tensor_ptr<int64_t>(
        {static_cast<int>(batchSizeActual)}, 
        std::move(labels)
    );
    
    return std::make_pair(inputTensor, labelTensor);
}

void SwiftDatasetAdapter::reset() {
    if (resetFunc) {
        resetFunc();
    }
}

void SwiftDatasetAdapter::setShuffle(bool shuffle) {
    if (setShuffleFunc) {
        setShuffleFunc(shuffle);
    }
}

size_t SwiftDatasetAdapter::size() const {
    if (sizeFunc) {
        return sizeFunc();
    }
    return 0;
}

size_t SwiftDatasetAdapter::inputDim() const {
    if (inputDimFunc) {
        return inputDimFunc();
    }
    return 0;
}

size_t SwiftDatasetAdapter::labelDim() const {
    if (labelDimFunc) {
        return labelDimFunc();
    }
    return 1; // Default to single label
} 