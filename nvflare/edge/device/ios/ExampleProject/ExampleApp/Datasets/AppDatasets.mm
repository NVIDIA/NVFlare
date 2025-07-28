//
//  AppDatasets.cpp
//  ExampleApp
//
//  Real app dataset implementations - ported from concrete implementation
//

#include "AppDatasets.hpp"
#include <executorch/extension/tensor/tensor.h>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <random>

// Objective-C++ imports for bundle loading and logging
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

using namespace executorch::extension;

namespace {
    // Helper function to load CIFAR-10 batch and normalize pixel values to [0,1]
    std::vector<CIFARImage> load_cifar10_batch(std::istream& dataStream, size_t maxImages) {
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
}

// AppCIFAR10Dataset implementation - real CIFAR10 data loading
AppCIFAR10Dataset::AppCIFAR10Dataset(std::istream& dataStream, bool shuffle) 
    : shouldShuffle(shuffle), currentIndex(0) {
    NSLog(@"AppCIFAR10Dataset: Loading real CIFAR10 data from stream");
    images = load_cifar10_batch(dataStream, maxImages);
    indices.resize(images.size());
    std::iota(indices.begin(), indices.end(), 0);
    reset();
}

// Default constructor for C interface - loads from app bundle
AppCIFAR10Dataset::AppCIFAR10Dataset() : shouldShuffle(false), currentIndex(0) {
    NSLog(@"AppCIFAR10Dataset: Default constructor - attempting to load from app bundle");
    
    // Try to load CIFAR-10 data from app bundle
    if (@available(iOS 9.0, *)) {
        NSDataAsset *dataAsset = [[NSDataAsset alloc] initWithName:@"data_batch_1"];
        if (dataAsset) {
            NSData *binaryData = dataAsset.data;
            const char *bytes = (const char *)[binaryData bytes];
            NSUInteger length = [binaryData length];
            std::istringstream dataStream(std::string(bytes, length));
            
            // Load data using the same function as the stream constructor
            images = load_cifar10_batch(dataStream, maxImages);
            indices.resize(images.size());
            std::iota(indices.begin(), indices.end(), 0);
            reset();
            
            NSLog(@"AppCIFAR10Dataset: Successfully loaded %lu CIFAR-10 images from app bundle", (unsigned long)images.size());
        } else {
            NSLog(@"AppCIFAR10Dataset: ERROR - No data_batch_1 found in app bundle. Dataset will be empty.");
            // No data available - dataset will have no images
        }
    } else {
        NSLog(@"AppCIFAR10Dataset: ERROR - NSDataAsset requires iOS 9.0+. Dataset will be empty.");
        // No data loading capability - dataset will have no images
    }
}

void AppCIFAR10Dataset::reset() {
    currentIndex = 0;
    if (shouldShuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }
}

void AppCIFAR10Dataset::setShuffle(bool shuffle) {
    shouldShuffle = shuffle;
    reset();
}

std::optional<AppCIFAR10Dataset::BatchType> AppCIFAR10Dataset::getBatch(size_t batchSize) {
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

size_t AppCIFAR10Dataset::size() const {
    return images.size();
}

size_t AppCIFAR10Dataset::inputDim() const {
    return cifar10::kImageSize;
}

size_t AppCIFAR10Dataset::labelDim() const {
    return cifar10::kLabelSize;
}

// AppXORDataset implementation - real XOR logic
AppXORDataset::AppXORDataset(bool shuffle) 
    : xor_table{{{1.0f, 1.0f}, 0},
                {{0.0f, 0.0f}, 0},
                {{1.0f, 0.0f}, 1},
                {{0.0f, 1.0f}, 1}},
      shouldShuffle(shuffle), currentIndex(0) {
    NSLog(@"AppXORDataset: Creating real XOR dataset with truth table");
    indices.resize(xor_table.size());
    std::iota(indices.begin(), indices.end(), 0);
    reset();
}

void AppXORDataset::reset() {
    currentIndex = 0;
    if (shouldShuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }
}

void AppXORDataset::setShuffle(bool shuffle) {
    shouldShuffle = shuffle;
    reset();
}

std::optional<AppXORDataset::BatchType> AppXORDataset::getBatch(size_t batchSize) {
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

size_t AppXORDataset::size() const {
    return xor_table.size();
}

size_t AppXORDataset::inputDim() const {
    return 2;
}

size_t AppXORDataset::labelDim() const {
    return 1;
}

// C interface functions for Swift to create app's C++ datasets
extern "C" {

/// Create app's CIFAR10 C++ dataset
void* CreateAppCIFAR10Dataset() {
            NSLog(@"C Interface: Creating AppCIFAR10Dataset");
    try {
        return new AppCIFAR10Dataset();
    } catch (const std::exception& e) {
        NSLog(@"C Interface: Failed to create AppCIFAR10Dataset: %s", e.what());
        return nullptr;
    }
}

/// Create app's XOR C++ dataset  
void* CreateAppXORDataset() {
    NSLog(@"🔍 C Interface: Creating AppXORDataset");
    try {
        AppXORDataset* dataset = new AppXORDataset();
        NSLog(@"🔍 C Interface: AppXORDataset created successfully at %p", dataset);
        
        // Test the object immediately after creation
        try {
            size_t testSize = dataset->size();
            NSLog(@"🔍 C Interface: AppXORDataset size test successful: %zu", testSize);
        } catch (const std::exception& e) {
            NSLog(@"❌ C Interface: AppXORDataset size test failed: %s", e.what());
            delete dataset;
            return nullptr;
        } catch (...) {
            NSLog(@"❌ C Interface: AppXORDataset size test failed with unknown exception");
            delete dataset;
            return nullptr;
        }
        
        return dataset;
    } catch (const std::exception& e) {
        NSLog(@"❌ C Interface: Failed to create AppXORDataset: %s", e.what());
        return nullptr;
    } catch (...) {
        NSLog(@"❌ C Interface: Failed to create AppXORDataset with unknown exception");
        return nullptr;
    }
}

/// Destroy app's C++ dataset
void DestroyAppDataset(void* dataset) {
    if (dataset) {
        NSLog(@"C Interface: Destroying app's C++ dataset");
        delete static_cast<ETDataset*>(dataset);
    }
}

} // extern "C" 
