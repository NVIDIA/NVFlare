//
//  Dataset.hpp
//  NVFlareMobile
//

#pragma once

#include <vector>

template<typename InputType, typename LabelType>
class Dataset {
public:
    virtual ~Dataset() = default;
    
    // Generic batch type
    using BatchType = std::vector<std::pair<InputType, LabelType>>;
    
    // Core interface
    virtual BatchType getBatch(size_t batchSize) = 0;
    virtual size_t size() const = 0;
    virtual size_t inputDim() const = 0;
    virtual size_t labelDim() const = 0;
};

// Example of a generic vector dataset
template<typename T>
class SimpleVectorDataset : public Dataset<std::vector<T>, T> {
private:
    std::vector<std::pair<std::vector<T>, T>> data;
    
protected:
    // Make data accessible to derived classes
    std::vector<std::pair<std::vector<T>, T>>& getData() { return data; }
    
public:
    using BatchType = typename Dataset<std::vector<T>, T>::BatchType;
    
    BatchType getBatch(size_t batchSize) override {
        BatchType batch;
        for (size_t i = 0; i < batchSize && i < data.size(); i++) {
            batch.push_back(data[i]);
        }
        return batch;
    }
    
    size_t size() const override { return data.size(); }
    size_t inputDim() const override { return data.empty() ? 0 : data[0].first.size(); }
    size_t labelDim() const override { return 1; }
};
