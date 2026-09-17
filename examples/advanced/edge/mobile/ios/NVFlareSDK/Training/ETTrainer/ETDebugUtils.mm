//
//  ETDebugUtils.mm
//  NVFlareMobile
//

#import <Foundation/Foundation.h>
#include "ETDebugUtils.h"


// Helper function for recursive tensor printing
void printElements(const torch::executor::Tensor& tensor,
                  torch::executor::ArrayRef<int> sizes,
                  torch::executor::ArrayRef<int> strides,
                  const float* data_ptr,
                  std::vector<int>& indices,
                  int dim) {
    if (dim == sizes.size()) {
        // Base case: we have a full index, calculate the linear index
        int64_t linear_index = 0;
        for (int i = 0; i < sizes.size(); ++i) {
            linear_index += indices[i] * strides[i];
        }
        // Access the element in the raw data
        float value = data_ptr[linear_index];
        // Print the element
        NSString* indexStr = @"";
        for (int i = 0; i < indices.size(); ++i) {
            indexStr = [indexStr stringByAppendingFormat:@"%d%@", indices[i],
                       (i < indices.size() - 1) ? @", " : @""];
        }
        NSLog(@"arr[%@] = %f", indexStr, value);
        return;
    }
    
    // Recursive case: loop through the current dimension
    for (int64_t i = 0; i < sizes[dim]; ++i) {
        indices.push_back(i);
        printElements(tensor, sizes, strides, data_ptr, indices, dim + 1);
        indices.pop_back();
    }
}


void printTensorElements(const torch::executor::Tensor& tensor) {
    auto strides = tensor.strides();
    auto data_ptr = tensor.const_data_ptr<float>();
    auto sizes = tensor.sizes();
    
    std::vector<int> indices;
    printElements(tensor, sizes, strides, data_ptr, indices, 0);
}

void printMap(const std::map<executorch::aten::string_view, executorch::aten::Tensor>& map) {
    for (const auto& pair : map) {
        NSLog(@"Key: %s", pair.first.data());
        printTensorElements(pair.second);
    }
}

void printTensorDictionary(NSDictionary<NSString *, id> *dict, NSString* name) {
    NSLog(@"%@ Dictionary Contents ===============", name);
    for (NSString *key in dict) {
        NSLog(@"Tensor: %@", key);
        NSDictionary *tensorInfo = dict[key];
        
        NSArray *sizes = tensorInfo[@"sizes"];
        NSLog(@"  Sizes: %@", sizes);
        
        NSArray *strides = tensorInfo[@"strides"];
        NSLog(@"  Strides: %@", strides);
        
        NSArray *data = tensorInfo[@"data"];
        NSLog(@"  Data[%lu]: [", (unsigned long)data.count);
        // Print first few and last few elements
        for (int i = 0; i < data.count; i++) {
            NSLog(@"    [%d]: %@", i, data[i]);
        }
        NSLog(@"  ]");
    }
    NSLog(@"End %@ Dictionary Contents ===========", name);
}
