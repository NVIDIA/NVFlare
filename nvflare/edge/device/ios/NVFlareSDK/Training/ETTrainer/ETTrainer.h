//
//  ETTrainer.h
//  NVFlareMobile
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// Forward declarations for Swift protocols
@protocol NVFlareDataSource;
@class NVFlareContext;

@interface ETTrainer : NSObject

/// Primary initializer - accepts C++ dataset directly  
/// @param modelBase64 Base64-encoded ExecutorTorch model data received from server
/// @param meta Training configuration dictionary containing parameters like batch_size, learning_rate, num_epochs
/// @param cppDataset Non-owning pointer to C++ ETDataset - passed from ETTrainerExecutor, lifecycle managed by NVFlareRunner
/// @return ETTrainer instance or nil if initialization fails (invalid model, dataset loading failure, etc.)
- (nullable instancetype)initWithModelBase64:(NSString *)modelBase64
                                     meta:(NSDictionary<NSString *, id> *)meta
                                  dataset:(void *)cppDataset;

/// Executes training using ExecutorTorch and returns model weight differences
/// @return Dictionary containing weight differences to be sent back to federated learning server
- (NSDictionary<NSString *, id> *)train;

@end

NS_ASSUME_NONNULL_END
