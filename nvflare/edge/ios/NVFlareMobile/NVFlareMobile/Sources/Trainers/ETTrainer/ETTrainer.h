//
//  ETTrainer.h
//  NVFlareMobile
//


#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface ETTrainer : NSObject

- (instancetype)initWithModelBase64:(NSString *)modelBase64
                             meta:(NSDictionary<NSString *, id> *)meta;

// Returns weight differences as dictionary
- (NSDictionary<NSString *, id> *)train;

@end

NS_ASSUME_NONNULL_END
