#include <jni.h>
#include <string>
#include <memory>
#include <android/log.h>
#include "training/training_module.h"

#define LOG_TAG "JNITrainingModule"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using executorch::extension::training::TrainingModule;

extern "C" JNIEXPORT jlong JNICALL
Java_com_nvidia_nvflare_trainer_ETTrainer_nativeInitializeTrainingModule(
    JNIEnv* env, jobject thiz, jstring modelPath) {
    // Convert Java string to C++ string
    const char* model_path_cstr = env->GetStringUTFChars(modelPath, nullptr);
    std::string model_path(model_path_cstr);
    env->ReleaseStringUTFChars(modelPath, model_path_cstr);

    // TODO: Use the model_path to load the model and create a TrainingModule instance
    // This is a placeholder for real training integration
    try {
        // Example: create a TrainingModule (with dummy args for now)
        std::unique_ptr<TrainingModule> module = std::make_unique<TrainingModule>(nullptr);
        // Return the pointer as a jlong (handle)
        return reinterpret_cast<jlong>(module.release());
    } catch (const std::exception& e) {
        LOGE("Failed to initialize TrainingModule: %s", e.what());
        return 0L;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_nvidia_nvflare_trainer_ETTrainer_nativeDestroyTrainingModule(
    JNIEnv* env, jobject thiz, jlong moduleHandle) {
    // Convert the handle back to a TrainingModule pointer and delete it
    auto* module = reinterpret_cast<TrainingModule*>(moduleHandle);
    delete module;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_nvidia_nvflare_trainer_ETTrainer_nativeTrain(
    JNIEnv* env, jobject thiz, jlong moduleHandle, jstring methodName, jint epochs, jint batchSize, jfloat learningRate, jfloat momentum, jfloat weightDecay) {
    // Convert the handle back to a TrainingModule pointer
    auto* module = reinterpret_cast<TrainingModule*>(moduleHandle);
    if (!module) {
        LOGE("Invalid module handle");
        return JNI_FALSE;
    }

    // Convert Java string to C++ string
    const char* method_name_cstr = env->GetStringUTFChars(methodName, nullptr);
    std::string method_name(method_name_cstr);
    env->ReleaseStringUTFChars(methodName, method_name_cstr);

    // TODO: Implement real training logic using module->execute_forward_backward(...)
    // This is a placeholder for real training integration
    try {
        // Example: call execute_forward_backward (with dummy inputs for now)
        std::vector<executorch::runtime::EValue> inputs;
        auto result = module->execute_forward_backward(method_name, inputs);
        if (result.ok()) {
            return JNI_TRUE;
        } else {
            LOGE("Training failed: %s", result.error().c_str());
            return JNI_FALSE;
        }
    } catch (const std::exception& e) {
        LOGE("Exception during training: %s", e.what());
        return JNI_FALSE;
    }
}

// TODO: Add JNI functions to destroy the module and perform training, as needed. 