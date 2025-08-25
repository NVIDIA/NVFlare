package com.nvidia.nvflare.app.data

import android.util.Log
import com.nvidia.nvflare.sdk.core.Context
import com.nvidia.nvflare.sdk.core.DataSource
import com.nvidia.nvflare.sdk.core.Dataset
import com.nvidia.nvflare.sdk.core.Batch
import com.nvidia.nvflare.app.data.XORDataset
import com.nvidia.nvflare.app.data.CIFAR10Dataset
import android.content.Context as AndroidContext

/**
 * Android-specific implementation of DataSource.
 * Provides datasets for training based on the dataset type.
 */
class AndroidDataSource(private val context: AndroidContext) : DataSource {
    
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
        
        Log.d("AndroidDataSource", "Job: $datasetType -> Dataset: $datasetName, Phase: $phase")
        
        return when (datasetName.lowercase()) {
            "cifar10" -> CIFAR10Dataset(context, phase)
            "xor" -> XORDataset(phase)
            else -> throw IllegalArgumentException("Unsupported dataset name: $datasetName")
        }
    }
}
