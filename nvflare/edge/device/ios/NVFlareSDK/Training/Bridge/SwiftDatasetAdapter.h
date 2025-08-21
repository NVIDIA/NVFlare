//
//  SwiftDatasetAdapter.h
//  NVFlareSDK
//
//  C++ adapter to bridge Swift datasets to ETDataset interface
//

#pragma once

#include "ETDataset.h"
#include <optional>
#include <vector>

/// C++ adapter that wraps Swift dataset callbacks
/// This allows Swift datasets to be used with the C++ ETTrainer
class SwiftDatasetAdapter : public ETDataset {
private:
    // Swift object reference passed in via CFBridgingRetain() on the Swift side.
    // Ownership is transferred to this adapter, which releases it via CFBridgingRelease().
    // DO NOT pass unretained (__bridge) Swift objects here.
    void* swiftObjectPtr;

    // Flag to prevent double deletion
    // Note: Not thread-safe; access must be synchronized if used concurrently.
    mutable bool isDestroyed;

public:
    /// Constructor with Swift object
    /// @param swiftObject Pointer from CFBridgingRetain - takes ownership
    SwiftDatasetAdapter(void* swiftObject);

    ~SwiftDatasetAdapter();

    // Required ETDataset interface methods
    std::optional<BatchType> getBatch(size_t batchSize) override;
    void reset() override;
    void setShuffle(bool shuffle) override;
    size_t size() const override;
    size_t inputDim() const override;
    size_t labelDim() const override;
};
