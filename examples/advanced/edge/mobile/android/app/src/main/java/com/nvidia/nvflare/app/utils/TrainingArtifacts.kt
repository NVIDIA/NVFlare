package com.nvidia.nvflare.app.utils

import android.content.Context
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

/**
 * Utility class to help access and manage training artifacts.
 * Provides easy access to saved models, logs, and training data.
 * Moved to app layer as it's application-specific functionality.
 */
class TrainingArtifacts(private val context: Context) {
    private val TAG = "TrainingArtifacts"
    
    /**
     * Get all training artifact directories.
     */
    fun getAllArtifactDirectories(): List<File> {
        val filesDir = context.filesDir
        return filesDir.listFiles()
            ?.filter { it.isDirectory && it.name.startsWith("training_artifacts_") }
            ?.sortedByDescending { it.lastModified() } // Most recent first
            ?: emptyList()
    }
    
    /**
     * Get the most recent training artifacts directory.
     */
    fun getLatestArtifactsDirectory(): File? {
        return getAllArtifactDirectories().firstOrNull()
    }
    
    /**
     * Get artifacts directory by timestamp.
     */
    fun getArtifactsDirectory(timestamp: String): File? {
        val artifactsDir = File(context.filesDir, "training_artifacts_$timestamp")
        return if (artifactsDir.exists()) artifactsDir else null
    }
    
    /**
     * List all saved models in an artifacts directory.
     */
    fun listSavedModels(artifactsDir: File): List<File> {
        val modelsDir = File(artifactsDir, "models")
        return if (modelsDir.exists()) {
            modelsDir.listFiles()?.toList() ?: emptyList()
        } else {
            emptyList()
        }
    }
    
    /**
     * List all log files in an artifacts directory.
     */
    fun listLogFiles(artifactsDir: File): List<File> {
        val logsDir = File(artifactsDir, "logs")
        return if (logsDir.exists()) {
            logsDir.listFiles()?.toList() ?: emptyList()
        } else {
            emptyList()
        }
    }
    
    /**
     * Get training summary from an artifacts directory.
     */
    fun getTrainingSummary(artifactsDir: File): String? {
        val summaryFile = File(artifactsDir, "logs/training_summary.txt")
        return if (summaryFile.exists()) {
            summaryFile.readText()
        } else {
            null
        }
    }
    
    /**
     * Get training progress from an artifacts directory.
     */
    fun getTrainingProgress(artifactsDir: File): String? {
        val progressFile = File(artifactsDir, "logs/training_progress.txt")
        return if (progressFile.exists()) {
            progressFile.readText()
        } else {
            null
        }
    }
    
    /**
     * Get initial model parameters from an artifacts directory.
     */
    fun getInitialParameters(artifactsDir: File): String? {
        val paramsFile = File(artifactsDir, "models/initial_parameters.json")
        return if (paramsFile.exists()) {
            paramsFile.readText()
        } else {
            null
        }
    }
    
    /**
     * Get final model parameters from an artifacts directory.
     */
    fun getFinalParameters(artifactsDir: File): String? {
        val paramsFile = File(artifactsDir, "models/final_parameters.json")
        return if (paramsFile.exists()) {
            paramsFile.readText()
        } else {
            null
        }
    }
    
    /**
     * Get tensor differences from an artifacts directory.
     */
    fun getTensorDifferences(artifactsDir: File): String? {
        val diffFile = File(artifactsDir, "models/tensor_differences.json")
        return if (diffFile.exists()) {
            diffFile.readText()
        } else {
            null
        }
    }
    
    /**
     * Get the initial model file (.pte) from an artifacts directory.
     */
    fun getInitialModelFile(artifactsDir: File): File? {
        val modelFile = File(artifactsDir, "models/initial_model.pte")
        return if (modelFile.exists()) modelFile else null
    }
    
    /**
     * Print a summary of all training artifacts.
     */
    fun printArtifactsSummary() {
        val artifacts = getAllArtifactDirectories()
        
        if (artifacts.isEmpty()) {
            Log.i(TAG, "No training artifacts found")
            return
        }
        
        Log.i(TAG, "Found ${artifacts.size} training artifact directories:")
        
        artifacts.forEach { artifactsDir ->
            val timestamp = artifactsDir.name.removePrefix("training_artifacts_")
            val date = try {
                val sdf = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
                val parsedDate = sdf.parse(timestamp)
                SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(parsedDate)
            } catch (e: Exception) {
                timestamp
            }
            
            Log.i(TAG, "  - $date (${artifactsDir.name})")
            Log.i(TAG, "    Location: ${artifactsDir.absolutePath}")
            
            // List models
            val models = listSavedModels(artifactsDir)
            if (models.isNotEmpty()) {
                Log.i(TAG, "    Models: ${models.joinToString(", ") { it.name }}")
            }
            
            // List logs
            val logs = listLogFiles(artifactsDir)
            if (logs.isNotEmpty()) {
                Log.i(TAG, "    Logs: ${logs.joinToString(", ") { it.name }}")
            }
            
            // Get summary
            val summary = getTrainingSummary(artifactsDir)
            if (summary != null) {
                val lines = summary.lines()
                val methodLine = lines.find { it.contains("Method:") }
                val epochsLine = lines.find { it.contains("Epochs:") }
                val lossLine = lines.find { it.contains("Average Loss:") }
                
                if (methodLine != null) Log.i(TAG, "    ${methodLine.trim()}")
                if (epochsLine != null) Log.i(TAG, "    ${epochsLine.trim()}")
                if (lossLine != null) Log.i(TAG, "    ${lossLine.trim()}")
            }
            
            Log.i(TAG, "")
        }
    }
    
    /**
     * Clean up old training artifacts (keep only the last N sessions).
     */
    fun cleanupOldArtifacts(keepLast: Int = 5) {
        val artifacts = getAllArtifactDirectories()
        
        if (artifacts.size <= keepLast) {
            Log.i(TAG, "No cleanup needed. Found ${artifacts.size} artifacts, keeping last $keepLast")
            return
        }
        
        val toDelete = artifacts.drop(keepLast)
        Log.i(TAG, "Cleaning up ${toDelete.size} old training artifacts")
        
        toDelete.forEach { artifactsDir ->
            try {
                artifactsDir.deleteRecursively()
                Log.i(TAG, "Deleted: ${artifactsDir.name}")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to delete: ${artifactsDir.name}", e)
            }
        }
    }
    
    /**
     * Export artifacts to external storage (if available).
     */
    fun exportArtifactsToExternal(artifactsDir: File, externalDir: File): Boolean {
        return try {
            if (!externalDir.exists()) {
                externalDir.mkdirs()
            }
            
            val timestamp = artifactsDir.name.removePrefix("training_artifacts_")
            val exportDir = File(externalDir, "nvflare_training_$timestamp")
            
            artifactsDir.copyRecursively(exportDir, overwrite = true)
            
            Log.i(TAG, "Exported artifacts to: ${exportDir.absolutePath}")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to export artifacts", e)
            false
        }
    }
    
    /**
     * Get a human-readable summary of training results.
     */
    fun getHumanReadableSummary(artifactsDir: File): String {
        val summary = getTrainingSummary(artifactsDir)
        val progress = getTrainingProgress(artifactsDir)
        
        return buildString {
            appendLine("Training Session Summary")
            appendLine("======================")
            appendLine("Directory: ${artifactsDir.name}")
            appendLine("Location: ${artifactsDir.absolutePath}")
            appendLine()
            
            if (summary != null) {
                // Extract key information from summary
                val lines = summary.lines()
                val methodLine = lines.find { it.contains("Method:") }
                val epochsLine = lines.find { it.contains("Epochs:") }
                val batchSizeLine = lines.find { it.contains("Batch Size:") }
                val lrLine = lines.find { it.contains("Learning Rate:") }
                val stepsLine = lines.find { it.contains("Total Steps:") }
                val lossLine = lines.find { it.contains("Average Loss:") }
                
                if (methodLine != null) appendLine(methodLine.trim())
                if (epochsLine != null) appendLine(epochsLine.trim())
                if (batchSizeLine != null) appendLine(batchSizeLine.trim())
                if (lrLine != null) appendLine(lrLine.trim())
                if (stepsLine != null) appendLine(stepsLine.trim())
                if (lossLine != null) appendLine(lossLine.trim())
            }
            
            appendLine()
            appendLine("Files Available:")
            
            val models = listSavedModels(artifactsDir)
            if (models.isNotEmpty()) {
                appendLine("  Models:")
                models.forEach { appendLine("    - ${it.name}") }
            }
            
            val logs = listLogFiles(artifactsDir)
            if (logs.isNotEmpty()) {
                appendLine("  Logs:")
                logs.forEach { appendLine("    - ${it.name}") }
            }
            
            if (progress != null) {
                appendLine()
                appendLine("Training Progress (last 10 entries):")
                progress.lines().takeLast(10).forEach { appendLine("  $it") }
            }
        }
    }
}
