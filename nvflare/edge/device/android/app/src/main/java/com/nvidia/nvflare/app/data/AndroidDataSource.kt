package com.nvidia.nvflare.app.data

import com.nvidia.nvflare.sdk.defs.Context
import com.nvidia.nvflare.sdk.defs.DataSource
import com.nvidia.nvflare.sdk.defs.Dataset
import com.nvidia.nvflare.sdk.defs.Batch
import com.nvidia.nvflare.app.data.XORDataset
import com.nvidia.nvflare.app.data.CIFAR10Dataset
import android.content.Context as AndroidContext

/**
 * Android-specific implementation of DataSource.
 * Provides datasets for training based on the dataset type.
 */
class AndroidDataSource(private val context: AndroidContext) : DataSource {
    
    override fun getDataset(datasetType: String, ctx: Context): Dataset {
        // datasetType should be one of: "train", "evaluate", "test"
        val phase = datasetType.lowercase()
        if (phase !in listOf("train", "evaluate", "test")) {
            throw IllegalArgumentException("Unsupported dataset phase: $datasetType. Must be one of: train, evaluate, test")
        }
        
        // Get the actual dataset name from context (e.g., "cifar10", "xor")
        val datasetName = ctx.get("dataset_name") as? String ?: "xor" // Default to XOR if not specified
        
        return when (datasetName.lowercase()) {
            "cifar10" -> CIFAR10Dataset(context, phase)
            "xor" -> XORDataset(phase)
            else -> throw IllegalArgumentException("Unsupported dataset name: $datasetName")
        }
    }
}
