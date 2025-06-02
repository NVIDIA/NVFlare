#include "ETDataset.hpp"
#include <executorch/extension/tensor/tensor.h>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <random>

using namespace executorch::extension;

// Helper function to load CIFAR-10 batch and normalize pixel values to [0,1]
static std::vector<CIFARImage> load_cifar10_batch(std::istream& dataStream, size_t maxImages) {
    std::vector<CIFARImage> dataset;
    
    // Calculate number of images from file size
    dataStream.seekg(0, std::ios::end);
    std::streamsize fileSize = dataStream.tellg();
    dataStream.seekg(0, std::ios::beg);
    
    if (fileSize <= 0) {
        return dataset;
    }
    
    // Calculate number of complete images in the file
    size_t numImages = fileSize / cifar10::kBytesPerImage;
    
    if (numImages == 0) {
        return dataset;
    }
    
    numImages = std::min(numImages, maxImages);
    
    dataset.reserve(numImages);
    
    for (size_t i = 0; i < numImages; i++) {
        CIFARImage image;
        image.data.resize(cifar10::kImageSize);
        uint8_t raw_data[cifar10::kImageSize];
        uint8_t label;

        // Read label and data from stream
        dataStream.read(reinterpret_cast<char*>(&label), cifar10::kLabelSize);
        dataStream.read(reinterpret_cast<char*>(raw_data), cifar10::kImageSize);
        
        if (dataStream.fail()) {
            break;
        }

        // Process data: normalize to [0,1]
        for (int j = 0; j < cifar10::kImageSize; j++) {
            image.data[j] = static_cast<float>(raw_data[j]) / 255.0f;
        }
        image.label = static_cast<int64_t>(label);

        dataset.push_back(std::move(image));
    }
    
    return dataset;
}

// CIFAR10Dataset implementation
CIFAR10Dataset::CIFAR10Dataset(std::istream& dataStream, bool shuffle) 
    : shouldShuffle(shuffle) {
    images = load_cifar10_batch(dataStream, maxImages);
    indices.resize(images.size());
    std::iota(indices.begin(), indices.end(), 0);
    reset();
}

void CIFAR10Dataset::reset() {
    currentIndex = 0;
    if (shouldShuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }
}

void CIFAR10Dataset::setShuffle(bool shuffle) {
    shouldShuffle = shuffle;
    reset();
}

std::optional<CIFAR10Dataset::BatchType> CIFAR10Dataset::getBatch(size_t batchSize) {
    if (currentIndex >= images.size()) {
        return std::nullopt;
    }

    size_t endIdx = std::min(currentIndex + batchSize, images.size());
    size_t actualBatchSize = endIdx - currentIndex;
    
    std::vector<float> image_batch;
    std::vector<int64_t> label_batch;
    image_batch.reserve(actualBatchSize * cifar10::kImageSize);
    label_batch.reserve(actualBatchSize);
    
    for (size_t i = currentIndex; i < endIdx; i++) {
        const auto& image = images[indices[i]];
        image_batch.insert(image_batch.end(),
                         image.data.begin(),
                         image.data.end());
        label_batch.push_back(image.label);
    }
    
    currentIndex = endIdx;
    return std::make_pair(
        make_tensor_ptr<float>({static_cast<int>(actualBatchSize), 
                              cifar10::kChannels,
                              cifar10::kImageWidth, 
                              cifar10::kImageHeight}, 
                             std::move(image_batch)),
        make_tensor_ptr<int64_t>({static_cast<int>(actualBatchSize)}, 
                               std::move(label_batch))
    );
}

size_t CIFAR10Dataset::size() const {
    return images.size();
}

size_t CIFAR10Dataset::inputDim() const {
    return cifar10::kImageSize;
}

size_t CIFAR10Dataset::labelDim() const {
    return cifar10::kLabelSize;
}

// XORDataset implementation
XORDataset::XORDataset(bool shuffle) 
    : xor_table{{{1.0f, 1.0f}, 0},
                {{0.0f, 0.0f}, 0},
                {{1.0f, 0.0f}, 1},
                {{0.0f, 1.0f}, 1}},
      shouldShuffle(shuffle) {
    indices.resize(xor_table.size());
    std::iota(indices.begin(), indices.end(), 0);
    reset();
}

void XORDataset::reset() {
    currentIndex = 0;
    if (shouldShuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }
}

void XORDataset::setShuffle(bool shuffle) {
    shouldShuffle = shuffle;
    reset();
}

std::optional<XORDataset::BatchType> XORDataset::getBatch(size_t batchSize) {
    if (currentIndex >= xor_table.size()) {
        return std::nullopt;
    }

    size_t endIdx = std::min(currentIndex + batchSize, xor_table.size());
    size_t actualBatchSize = endIdx - currentIndex;
    
    std::vector<float> input_batch;
    std::vector<int64_t> label_batch;
    input_batch.reserve(actualBatchSize * 2); // 2 features per sample
    label_batch.reserve(actualBatchSize);
    
    for (size_t i = currentIndex; i < endIdx; i++) {
        const auto& [inputs, label] = xor_table[indices[i]];
        input_batch.insert(input_batch.end(), inputs.begin(), inputs.end());
        label_batch.push_back(label);
    }
    
    currentIndex = endIdx;
    return std::make_pair(
        make_tensor_ptr<float>({static_cast<int>(actualBatchSize), 2}, 
                             std::move(input_batch)),
        make_tensor_ptr<int64_t>({static_cast<int>(actualBatchSize)}, 
                               std::move(label_batch))
    );
}

size_t XORDataset::size() const {
    return xor_table.size();
}

size_t XORDataset::inputDim() const {
    return 2;
}

size_t XORDataset::labelDim() const {
    return 1;
}
