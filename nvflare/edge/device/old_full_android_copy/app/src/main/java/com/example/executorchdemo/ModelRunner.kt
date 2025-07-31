/*
package com.example.executorchdemo

import android.content.Context
import android.util.Log
import org.pytorch.executorch.Module
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream

class ModelRunner(private val context: Context) {
    private var module: Module? = null
    private val TAG = "ModelRunner"
    private val MODEL_FILE_NAME = "add_model.pte"

    fun loadModel() {
        try {
            Log.d(TAG, "Starting model loading process...")
            val modelPath = copyModelFromAssets()
            Log.d(TAG, "Attempting to load module from path: $modelPath")
            module = Module.load(modelPath)
            Log.d(TAG, "Model loaded successfully from path: $modelPath")
        } catch (e: IOException) {
            Log.e(TAG, "Error loading model: ${e.message}", e)
            throw RuntimeException("Failed to load model", e)
        } catch (e: Exception) {
            Log.e(TAG, "Unexpected error loading model: ${e.message}", e)
            throw RuntimeException("Unexpected error loading model", e)
        }
    }

    private fun copyModelFromAssets(): String {
        val modelFile = File(context.filesDir, MODEL_FILE_NAME)
        Log.d(TAG, "Checking model file at: ${modelFile.absolutePath}")
        
        if (!modelFile.exists()) {
            try {
                Log.d(TAG, "Model file not found, copying from assets...")
                context.assets.open("models/$MODEL_FILE_NAME").use { inputStream ->
                    FileOutputStream(modelFile).use { outputStream ->
                        val buffer = ByteArray(4 * 1024)
                        var read: Int
                        while (inputStream.read(buffer).also { read = it } != -1) {
                            outputStream.write(buffer, 0, read)
                        }
                    }
                }
                Log.d(TAG, "Model file copied successfully to: ${modelFile.absolutePath}")
            } catch (e: IOException) {
                Log.e(TAG, "Error copying model file: ${e.message}", e)
                throw RuntimeException("Failed to copy model file", e)
            }
        } else {
            Log.d(TAG, "Model file already exists at: ${modelFile.absolutePath}")
        }
        
        return modelFile.absolutePath
    }

    fun runModel(input1: Float, input2: Float): Float {
        if (module == null) {
            Log.e(TAG, "Module is null! Make sure to call loadModel() first.")
            throw IllegalStateException("Model not loaded. Call loadModel() first.")
        }

        return try {
            Log.d(TAG, "Starting model inference with inputs: ($input1, $input2)")
            
            // Create two 1D tensors with one element each, as expected by the model
            val input1Data = floatArrayOf(input1)
            val input2Data = floatArrayOf(input2)
            Log.d(TAG, "Created input data arrays: [${input1Data[0]}], [${input2Data[0]}]")
            
            val input1Tensor = Tensor.fromBlob(input1Data, longArrayOf(1))
            val input2Tensor = Tensor.fromBlob(input2Data, longArrayOf(1))
            Log.d(TAG, "Created input tensors with shape [1]")
            
            // Create EValues from tensors
            val inputValue1 = EValue.from(input1Tensor)
            val inputValue2 = EValue.from(input2Tensor)
            Log.d(TAG, "Created EValues from tensors")
            
            Log.d(TAG, "Running model forward pass...")
            val outputs = module?.forward(inputValue1, inputValue2)
            Log.d(TAG, "Model forward pass completed")
            
            val result = outputs?.get(0)?.toDouble()?.toFloat() ?: throw IllegalStateException("Model returned null result")
            Log.d(TAG, "Model inference successful. Input: ($input1, $input2), Output: $result")
            
            result
        } catch (e: Exception) {
            Log.e(TAG, "Error running model: ${e.message}", e)
            e.printStackTrace()
            throw RuntimeException("Failed to run model inference", e)
        }
    }
} */
