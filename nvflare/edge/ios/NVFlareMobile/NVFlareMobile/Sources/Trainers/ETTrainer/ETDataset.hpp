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


struct CIFARImage {
    int64_t label;
    std::vector<float> data;  // Store image data as float
};

// ETTrainer specific dataset using TensorPtr
using ETDataset = Dataset<executorch::extension::TensorPtr, executorch::extension::TensorPtr>;

class CIFAR10Dataset : public ETDataset {
private:
    std::vector<CIFARImage> images;
    
public:
    explicit CIFAR10Dataset(std::istream& dataStream);
    
    BatchType getBatch(size_t batchSize) override;
    size_t size() const override;
    size_t inputDim() const override;
    size_t labelDim() const override;
};

class XORDataset : public ETDataset {
private:
    std::vector<std::pair<std::vector<float>, int64_t>> xor_table;
    
public:
    XORDataset();
    
    BatchType getBatch(size_t batchSize) override;
    size_t size() const override;
    size_t inputDim() const override;
    size_t labelDim() const override;
};
