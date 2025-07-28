//
//  AppDatasets.hpp  
//  ExampleApp
//
//  Real app dataset implementations extending SDK's ETDataset base class
//

#pragma once

#include "ETDataset.h"
#include <istream>
#include <array>

// CIFAR-10 format constants for this app
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

/// App's CIFAR10 dataset - real implementation with data loading
class AppCIFAR10Dataset : public ETDataset {
private:
    std::vector<CIFARImage> images;
    std::vector<size_t> indices;
    bool shouldShuffle;
    size_t currentIndex;
    size_t maxImages = 16;  // Demo limit
    
public:
    explicit AppCIFAR10Dataset(std::istream& dataStream, bool shuffle = false);
    AppCIFAR10Dataset();  // Default constructor for C interface
    
    // Required ETDataset interface
    std::optional<BatchType> getBatch(size_t batchSize) override;
    void reset() override;
    void setShuffle(bool shuffle) override;
    size_t size() const override;
    size_t inputDim() const override;
    size_t labelDim() const override;
};

/// App's XOR dataset - real implementation with XOR logic
class AppXORDataset : public ETDataset {
private:
    const std::vector<std::pair<std::array<float, 2>, int64_t>> xor_table;
    std::vector<size_t> indices;
    bool shouldShuffle;
    size_t currentIndex;
    
public:
    explicit AppXORDataset(bool shuffle = false);
    
    // Required ETDataset interface
    std::optional<BatchType> getBatch(size_t batchSize) override;
    void reset() override;
    void setShuffle(bool shuffle) override;
    size_t size() const override;
    size_t inputDim() const override;
    size_t labelDim() const override;
}; 
