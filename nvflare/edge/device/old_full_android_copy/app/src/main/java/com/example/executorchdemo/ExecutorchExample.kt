/*
package com.example.executorchdemo

import org.pytorch.executorch.Module
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor
import android.content.Context
import android.util.Log
import java.io.File

class ExecutorchExample(private val context: Context) {
    private val TAG = "ExecutorchExample"

    fun runSimpleAddition(): Float {
        try {
            Log.d(TAG, "Starting runSimpleAddition")
            
            // Create two 1D tensors with one element each, as expected by the model
            val input1Data = floatArrayOf(1.0f)
            val input2Data = floatArrayOf(1.0f)
            Log.d(TAG, "Created input data arrays: [${input1Data[0]}], [${input2Data[0]}]")
            
            val input1Tensor = Tensor.fromBlob(input1Data, longArrayOf(1))
            val input2Tensor = Tensor.fromBlob(input2Data, longArrayOf(1))
            Log.d(TAG, "Created input tensors with shape [1]")
            
            // Create EValues from tensors
            val input1 = EValue.from(input1Tensor)
            val input2 = EValue.from(input2Tensor)
            Log.d(TAG, "Created EValues from tensors")
            
            // Load the model
            val modelFile = File(context.filesDir, "add_model.pte")
            Log.d(TAG, "Loading model from: ${modelFile.absolutePath}")
            val module = Module.load(modelFile.absolutePath)
            Log.d(TAG, "Model loaded successfully")
            
            // Run the model with both inputs
            Log.d(TAG, "Running model forward pass")
            val outputs = module.forward(input1, input2)
            Log.d(TAG, "Model forward pass completed")
            
            // Get the result tensor and extract the scalar value
            val resultTensor = outputs[0].toTensor()
            val result = resultTensor.dataAsFloatArray[0]
            Log.d(TAG, "Got result: $result")
            
            return result
        } catch (e: Exception) {
            Log.e(TAG, "Error in runSimpleAddition: ${e.message}", e)
            e.printStackTrace()
            throw e
        }
    }
} */
