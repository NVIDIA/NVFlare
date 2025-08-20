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
/// @return ETTrainer instance or nil if initialization fails (invalid model, dataset loading failure, etc.)
- (nullable instancetype)initWithModelBase64:(NSString *)modelBase64
                                     meta:(NSDictionary<NSString *, id> *)meta
                                  dataset:(void *)cppDataset;

/// Returns weight differences as dictionary
- (NSDictionary<NSString *, id> *)train;

@end

NS_ASSUME_NONNULL_END
