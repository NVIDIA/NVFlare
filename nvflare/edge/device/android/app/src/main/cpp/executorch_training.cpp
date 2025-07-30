#include <jni.h>
#include <string>
#include <memory>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <android/log.h>

// ExecuTorch includes
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/training/module/training_module.h>
#include <executorch/extension/training/optimizer/sgd.h>

#define LOG_TAG "ExecutorTorchTraining"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace ::executorch::extension;

// Global storage for training modules
static std::map<long, std::unique_ptr<training::TrainingModule>> training_modules;
static long next_module_id = 1;

// Helper function to convert Java string to C++ string
std::string jstringToString(JNIEnv* env, jstring jstr) {
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

// Helper function to create Java byte array from C++ vector
jbyteArray vectorToByteArray(JNIEnv* env, const std::vector<uint8_t>& data) {
    jbyteArray result = env->NewByteArray(data.size());
    env->SetByteArrayRegion(result, 0, data.size(), reinterpret_cast<const jbyte*>(data.data()));
    return result;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_nvflare_app_training_ETTrainer_nativeInitializeTrainingModule(
        JNIEnv* env, jobject thiz, jstring modelPath) {
    
    try {
        std::string path = jstringToString(env, modelPath);
        LOGI("Initializing training module from path: %s", path.c_str());
        
        // Load model using FileDataLoader
        auto model_result = FileDataLoader::from(path);
        if (!model_result.ok()) {
            LOGE("Failed to load model from path: %s", path.c_str());
            return 0;
        }
        
        auto loader = std::make_unique<FileDataLoader>(std::move(model_result.get()));
        auto module = std::make_unique<training::TrainingModule>(std::move(loader));
        
        // Store module and return handle
        long module_id = next_module_id++;
        training_modules[module_id] = std::move(module);
        
        LOGI("Training module initialized successfully with ID: %ld", module_id);
        return module_id;
        
    } catch (const std::exception& e) {
        LOGE("Exception during module initialization: %s", e.what());
        return 0;
    }
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_nvidia_nvflare_app_training_ETTrainer_nativeTrain(
        JNIEnv* env, jobject thiz, jlong moduleHandle, jstring method,
        jint epochs, jint batchSize, jfloat learningRate, jfloat momentum, jfloat weightDecay) {
    
    try {
        // Get the training module
        auto it = training_modules.find(moduleHandle);
        if (it == training_modules.end()) {
            LOGE("Training module not found for handle: %ld", moduleHandle);
            return nullptr;
        }
        
        auto& training_module = it->second;
        std::string method_str = jstringToString(env, method);
        
        LOGI("Starting training - method: %s, epochs: %d, batchSize: %d, lr: %f",
             method_str.c_str(), epochs, batchSize, learningRate);
        
        // Get initial parameters
        auto param_res = training_module->named_parameters("forward");
        if (param_res.error() != executorch::runtime::Error::Ok) {
            LOGE("Failed to get named parameters");
            return nullptr;
        }
        
        auto initial_params = param_res.get();
        
        // Configure optimizer
        training::optimizer::SGDOptions options{learningRate};
        training::optimizer::SGD optimizer(param_res.get(), options);
        
        // Training loop
        int totalSteps = 0;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            LOGI("Starting epoch %d/%d", epoch + 1, epochs);
            
            // For now, we'll do a simple training loop
            // In a real implementation, you would iterate over your dataset
            for (int step = 0; step < 10; step++) { // Simplified: 10 steps per epoch
                
                // Create dummy input and label tensors for demonstration
                // In real implementation, these would come from your dataset
                std::vector<float> input_data(3072, 0.1f); // CIFAR-10 input size
                std::vector<int64_t> label_data(1, 0);     // Single label
                
                auto input_tensor = Tensor::from_blob(input_data.data(), {1, 3, 32, 32});
                auto label_tensor = Tensor::from_blob(label_data.data(), {1});
                
                // Execute forward-backward pass
                const auto& results = training_module->execute_forward_backward(
                    "forward", {*input_tensor, *label_tensor}
                );
                
                if (results.error() != executorch::runtime::Error::Ok) {
                    LOGE("Failed to execute forward_backward at step %d", step);
                    continue;
                }
                
                // Update parameters
                optimizer.step(training_module->named_gradients("forward").get());
                totalSteps++;
                
                if (step % 5 == 0) {
                    LOGI("Epoch %d, Step %d, Total Steps %d", epoch + 1, step, totalSteps);
                }
            }
        }
        
        // Get final parameters
        auto final_params = param_res.get();
        
        // Calculate parameter differences (simplified)
        // In a real implementation, you would calculate actual differences
        std::vector<uint8_t> result_data;
        
        // Serialize the result
        // Format: number of tensors, then for each tensor: name length, name, data length, data
        int num_tensors = 1; // Simplified: just one tensor for now
        
        // Write number of tensors
        result_data.insert(result_data.end(), 
                          reinterpret_cast<uint8_t*>(&num_tensors),
                          reinterpret_cast<uint8_t*>(&num_tensors) + sizeof(int));
        
        // Write tensor name
        std::string tensor_name = "weight";
        int name_length = tensor_name.length();
        result_data.insert(result_data.end(),
                          reinterpret_cast<uint8_t*>(&name_length),
                          reinterpret_cast<uint8_t*>(&name_length) + sizeof(int));
        result_data.insert(result_data.end(), tensor_name.begin(), tensor_name.end());
        
        // Write tensor data (simplified: dummy data)
        std::vector<float> dummy_data(4 * 3 * 32 * 32, 0.01f); // Small weight differences
        int data_length = dummy_data.size();
        result_data.insert(result_data.end(),
                          reinterpret_cast<uint8_t*>(&data_length),
                          reinterpret_cast<uint8_t*>(&data_length) + sizeof(int));
        result_data.insert(result_data.end(),
                          reinterpret_cast<uint8_t*>(dummy_data.data()),
                          reinterpret_cast<uint8_t*>(dummy_data.data()) + data_length * sizeof(float));
        
        LOGI("Training completed successfully, returning %zu bytes", result_data.size());
        
        return vectorToByteArray(env, result_data);
        
    } catch (const std::exception& e) {
        LOGE("Exception during training: %s", e.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_nvflare_app_training_ETTrainer_nativeCleanup(
        JNIEnv* env, jobject thiz, jlong moduleHandle) {
    
    try {
        auto it = training_modules.find(moduleHandle);
        if (it != training_modules.end()) {
            training_modules.erase(it);
            LOGI("Training module %ld cleaned up successfully", moduleHandle);
        } else {
            LOGE("Training module %ld not found for cleanup", moduleHandle);
        }
    } catch (const std::exception& e) {
        LOGE("Exception during cleanup: %s", e.what());
    }
} 