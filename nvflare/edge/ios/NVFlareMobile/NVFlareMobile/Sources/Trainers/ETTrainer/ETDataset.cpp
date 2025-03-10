#include "ETDataset.hpp"
#include <executorch/extension/tensor/tensor.h>
#include <sstream>

using namespace executorch::extension;

// Constants
const int IMAGE_SIZE = 32 * 32 * 3;  // 3072 bytes per image
const int NUM_IMAGES = 10000;        // Each batch has 10,000 images

// Helper function to load CIFAR-10 batch and normalize pixel values to [0,1]
static std::vector<CIFARImage> load_cifar10_batch(std::istream& dataStream) {
    std::vector<CIFARImage> dataset;
    
    for (int i = 0; i < NUM_IMAGES; i++) {
        CIFARImage image;
        image.data.resize(IMAGE_SIZE);
        uint8_t raw_data[IMAGE_SIZE];

        // Read label and data from stream
        dataStream.read(reinterpret_cast<char*>(&image.label), 1);
        dataStream.read(reinterpret_cast<char*>(raw_data), IMAGE_SIZE);

        // Process data: normalize to [0,1]
        for (int j = 0; j < IMAGE_SIZE; j++) {
            image.data[j] = static_cast<float>(raw_data[j]) / 255.0f;
        }
        image.label = static_cast<int64_t>(image.label);

        dataset.push_back(image);
    }
    
    return dataset;
}

// CIFAR10Dataset implementation
CIFAR10Dataset::CIFAR10Dataset(std::istream& dataStream) {
    images = load_cifar10_batch(dataStream);
}

CIFAR10Dataset::BatchType CIFAR10Dataset::getBatch(size_t batchSize) {
    BatchType batch;
    const int image_size = 3 * 32 * 32;  // channels * height * width
    
    // Prepare batch containers
    std::vector<float> image_batch;
    image_batch.reserve(batchSize * image_size);
    std::vector<int64_t> label_batch;
    label_batch.reserve(batchSize);
    
    // Fill batch with images and labels
    for (size_t i = 0; i < batchSize && i < images.size(); i++) {
        const auto& image = images[i];
        image_batch.insert(image_batch.end(),
                         image.data.begin(),
                         image.data.end());
        label_batch.push_back(image.label);
    }
    
    batch.push_back({
        make_tensor_ptr<float>({static_cast<int>(batchSize), 3, 32, 32}, image_batch),
        make_tensor_ptr<int64_t>({static_cast<int>(batchSize)}, label_batch)
    });
    
    return batch;
}

size_t CIFAR10Dataset::size() const {
    return images.size();
}

size_t CIFAR10Dataset::inputDim() const {
    return 3 * 32 * 32;  // channels * height * width
}

size_t CIFAR10Dataset::labelDim() const {
    return 1;  // Single class label
}

// XORDataset implementation
XORDataset::XORDataset() : xor_table{
    {{1.0f, 1.0f}, 0},
    {{0.0f, 0.0f}, 0},
    {{1.0f, 0.0f}, 1},
    {{0.0f, 1.0f}, 1}
} {}

XORDataset::BatchType XORDataset::getBatch(size_t batchSize) {
    BatchType batch;
    
    for (size_t i = 0; i < batchSize && i < xor_table.size(); i++) {
        const auto& [inputs, label] = xor_table[i];
        batch.push_back({
            make_tensor_ptr<float>({1, 2}, inputs),
            make_tensor_ptr<int64_t>({1}, {label})
        });
    }
    
    return batch;
}

size_t XORDataset::size() const {
    return xor_table.size();
}

size_t XORDataset::inputDim() const {
    return 2;  // Two input features for XOR
}

size_t XORDataset::labelDim() const {
    return 1;  // Binary output
}
