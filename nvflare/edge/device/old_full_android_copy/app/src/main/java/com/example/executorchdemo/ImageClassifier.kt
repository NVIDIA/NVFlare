/*
package com.example.executorchdemo

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.executorch.Module
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor
import java.io.File
import java.nio.FloatBuffer

class ImageClassifier(private val context: Context) {
    private val TAG = "ImageClassifier"
    private var module: Module? = null
    
    init {
        loadModel()
    }
    
    private fun loadModel() {
        try {
            val modelFile = File(context.filesDir, "image_classifier.pte")
            Log.d(TAG, "Loading model from: ${modelFile.absolutePath}")
            module = Module.load(modelFile.absolutePath)
            Log.d(TAG, "Model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model: ${e.message}", e)
            throw e
        }
    }
    
    fun classifyImage(bitmap: Bitmap): FloatArray {
        try {
            // Resize bitmap to 224x224 (MobileNetV2 input size)
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
            
            // Convert bitmap to float array (normalized to [0,1])
            val inputData = FloatArray(3 * 224 * 224)
            val pixels = IntArray(224 * 224)
            resizedBitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224)
            
            // MobileNetV2 expects RGB values normalized to [0,1]
            for (i in pixels.indices) {
                val pixel = pixels[i]
                inputData[i] = (pixel shr 16 and 0xFF) / 255.0f // R
                inputData[i + 224 * 224] = (pixel shr 8 and 0xFF) / 255.0f // G
                inputData[i + 2 * 224 * 224] = (pixel and 0xFF) / 255.0f // B
            }
            
            // Create input tensor
            val inputTensor = Tensor.fromBlob(inputData, longArrayOf(1, 3, 224, 224))
            val input = EValue.from(inputTensor)
            
            // Run inference
            val outputs = module!!.forward(input)
            val outputTensor = outputs[0].toTensor()
            
            // Get probabilities
            val probabilities = outputTensor.dataAsFloatArray
            
            return probabilities
        } catch (e: Exception) {
            Log.e(TAG, "Error in classifyImage: ${e.message}", e)
            throw e
        }
    }
} */
