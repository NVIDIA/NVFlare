//
//  ETDataset.hpp
//  NVFlareMobile
//
//

#pragma once

#include "../../Datasets/Dataset.hpp"
#include <executorch/extension/tensor/tensor.h>
#include <istream>
#include <vector>
#include <optional>

// CIFAR-10 format constants
namespace cifar10 {
    constexpr int kImageWidth = 32;
    constexpr int kImageHeight = 32;
    constexpr int kChannels = 3;
    constexpr int kImageSize = kImageWidth * kImageHeight * kChannels;  // 3072 bytes per image
    constexpr int kLabelSize = 1;  // Single class label
    constexpr int kBytesPerImage = kLabelSize + kImageSize;  // Total bytes per image entry
}

struct CIFARImage {
    int64_t label;
    std::vector<float> data;  // Store image data as float
};

// ETTrainer specific dataset using TensorPtr
using ETDataset = Dataset<executorch::extension::TensorPtr, executorch::extension::TensorPtr>;

class CIFAR10Dataset : public ETDataset {
private:
    std::vector<CIFARImage> images;
    std::vector<size_t> indices;
    bool shouldShuffle;
    size_t currentIndex;
    size_t maxImages = 16;
    
public:
    explicit CIFAR10Dataset(std::istream& dataStream, bool shuffle = false);
    
    std::optional<BatchType> getBatch(size_t batchSize) override;
    void reset() override;
    void setShuffle(bool shuffle) override;
    size_t size() const override;
    size_t inputDim() const override;
    size_t labelDim() const override;
};

class XORDataset : public ETDataset {
private:
    const std::vector<std::pair<std::array<float, 2>, int64_t>> xor_table;
    std::vector<size_t> indices;
    bool shouldShuffle;
    size_t currentIndex;
    
public:
    explicit XORDataset(bool shuffle = false);
    
    std::optional<BatchType> getBatch(size_t batchSize) override;
    void reset() override;
    void setShuffle(bool shuffle) override;
    size_t size() const override;
    size_t inputDim() const override;
    size_t labelDim() const override;
};
