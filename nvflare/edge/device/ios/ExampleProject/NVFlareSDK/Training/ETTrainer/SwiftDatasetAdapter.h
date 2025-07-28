//
//  SwiftDatasetAdapter.h
//  NVFlareSDK
//
//  C++ adapter to bridge Swift datasets to ETDataset interface
//

#pragma once

#include "ETDataset.h"
#include <functional>
#include <optional>
#include <vector>

/// C++ adapter that wraps Swift dataset callbacks
/// This allows Swift datasets to be used with the C++ ETTrainer
class SwiftDatasetAdapter : public ETDataset {
private:
    // Swift callback functions
    std::function<std::optional<std::pair<std::vector<float>, std::vector<int64_t>>>(size_t)> getBatchFunc;
    std::function<void()> resetFunc;
    std::function<size_t()> sizeFunc;
    std::function<size_t()> inputDimFunc;
    std::function<size_t()> labelDimFunc;
    std::function<void(bool)> setShuffleFunc;
    
public:
    /// Constructor with Swift callbacks
    SwiftDatasetAdapter(
        std::function<std::optional<std::pair<std::vector<float>, std::vector<int64_t>>>(size_t)> getBatch,
        std::function<void()> reset,
        std::function<size_t()> size,
        std::function<size_t()> inputDim,
        std::function<size_t()> labelDim,
        std::function<void(bool)> setShuffle
    );
    
    // Required ETDataset interface
    std::optional<BatchType> getBatch(size_t batchSize) override;
    void reset() override;
    void setShuffle(bool shuffle) override;
    size_t size() const override;
    size_t inputDim() const override;
    size_t labelDim() const override;
}; 