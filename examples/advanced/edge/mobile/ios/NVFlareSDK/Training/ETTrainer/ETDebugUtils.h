//
//  ETDebugUtils.h
//  NVFlareMobile
//

#pragma once

#import <Foundation/Foundation.h>

#ifdef __cplusplus
#include <executorch/extension/tensor/tensor.h>
#include <map>

// All functions are Objective-C++ since they use NSLog
extern "C" {

// Print tensor dictionary
void printTensorDictionary(NSDictionary<NSString *, id> *dict, NSString* name);

// Print tensor elements
void printTensorElements(const torch::executor::Tensor& tensor);

// Print tensor map
void printMap(const std::map<executorch::aten::string_view, executorch::aten::Tensor>& map);

}
#endif
