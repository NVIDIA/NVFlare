package com.nvidia.nvflare.app.data

import com.nvidia.nvflare.sdk.core.DataSource
import com.nvidia.nvflare.sdk.core.Context
import com.nvidia.nvflare.sdk.core.Dataset
import android.content.Context as AndroidContext

/**
 * Sample implementation of DataSource interface.
 * This shows users how to implement the DataSource interface to provide
 * their own datasets instead of having hardcoded dataset creation in the trainer.
 * 
 * Users should implement their own DataSource based on their specific needs.
 */
class SampleDataSource(private val androidContext: AndroidContext) : DataSource {
    
    override fun getDataset(datasetType: String, ctx: Context): Dataset {
        // datasetType is the job name (e.g., "xor_et", "cifar10_et")
        // Extract the actual dataset type from the job name
        val datasetName = when {
            datasetType.lowercase().contains("xor") -> "xor"
            datasetType.lowercase().contains("cifar") || datasetType.lowercase().contains("cnn") -> "cifar10"
            else -> "xor" // Default to XOR if unknown
        }
        
        // Always use "train" phase for federated learning
        val phase = "train"
        
        return when (datasetName.lowercase()) {
            "xor" -> {
                // Return XOR dataset for XOR training jobs
                XORDataset(phase)
            }
            "cifar10" -> {
                // Return CIFAR-10 dataset for CNN training jobs
                CIFAR10Dataset(androidContext, phase)
            }
            else -> {
                // Throw exception for unsupported dataset types
                throw IllegalArgumentException("Unsupported dataset type: $datasetType")
            }
        }
    }
}
