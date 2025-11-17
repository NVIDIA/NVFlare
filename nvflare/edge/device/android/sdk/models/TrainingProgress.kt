package com.nvidia.nvflare.sdk.models

/**
 * Represents the current state and progress of training.
 * Used to communicate training status from FlareRunner to UI.
 * 
 * This is part of the SDK to provide a standardized progress tracking model
 * that all apps using NVFlare SDK can use.
 */
data class TrainingProgress(
    val phase: TrainingPhase,
    val message: String,
    val currentRound: Int? = null,
    val totalRounds: Int? = null,
    val currentEpoch: Int? = null,
    val totalEpochs: Int? = null,
    val taskName: String? = null,
    val jobId: String? = null,
    val jobName: String? = null,
    val serverUrl: String? = null,
    val datasetSize: Int? = null,
    val errorDetails: String? = null,
    val duration: Long? = null,  // Duration in milliseconds
    val timestamp: Long = System.currentTimeMillis()
) {
    companion object {
        fun idle() = TrainingProgress(
            phase = TrainingPhase.IDLE,
            message = "Ready to start training"
        )
        
        fun connecting(serverUrl: String? = null) = TrainingProgress(
            phase = TrainingPhase.CONNECTING,
            message = "Connecting to server${serverUrl?.let { ": $it" } ?: ""}",
            serverUrl = serverUrl
        )
        
        fun fetchingJob(datasetSize: Int? = null) = TrainingProgress(
            phase = TrainingPhase.FETCHING_JOB,
            message = "Fetching job from server${datasetSize?.let { " (Dataset: $it samples)" } ?: ""}",
            datasetSize = datasetSize
        )
        
        fun jobReceived(jobId: String, jobName: String, duration: Long? = null) = TrainingProgress(
            phase = TrainingPhase.JOB_RECEIVED,
            message = "Job received: $jobName${duration?.let { " (${it}ms)" } ?: ""}",
            jobId = jobId,
            jobName = jobName,
            duration = duration
        )
        
        fun fetchingTask(currentRound: Int, totalRounds: Int) = TrainingProgress(
            phase = TrainingPhase.FETCHING_TASK,
            message = "Fetching task for round $currentRound/$totalRounds",
            currentRound = currentRound,
            totalRounds = totalRounds
        )
        
        fun taskReceived(taskName: String, currentRound: Int, totalRounds: Int) = TrainingProgress(
            phase = TrainingPhase.TASK_RECEIVED,
            message = "Task received: $taskName (Round $currentRound/$totalRounds)",
            taskName = taskName,
            currentRound = currentRound,
            totalRounds = totalRounds
        )
        
        fun training(taskName: String, currentRound: Int, totalRounds: Int, currentEpoch: Int, totalEpochs: Int, duration: Long? = null) = TrainingProgress(
            phase = TrainingPhase.TRAINING,
            message = "Training: Round $currentRound/$totalRounds, Epoch $currentEpoch/$totalEpochs${duration?.let { " (${it}ms)" } ?: ""}",
            taskName = taskName,
            currentRound = currentRound,
            totalRounds = totalRounds,
            currentEpoch = currentEpoch,
            totalEpochs = totalEpochs,
            duration = duration
        )
        
        fun sendingResults(taskName: String, currentRound: Int, totalRounds: Int) = TrainingProgress(
            phase = TrainingPhase.SENDING_RESULTS,
            message = "Sending results for round $currentRound/$totalRounds",
            taskName = taskName,
            currentRound = currentRound,
            totalRounds = totalRounds
        )
        
        fun resultsSent(currentRound: Int, totalRounds: Int, duration: Long? = null) = TrainingProgress(
            phase = TrainingPhase.RESULTS_SENT,
            message = "Results sent successfully (Round $currentRound/$totalRounds)${duration?.let { " (${it}ms)" } ?: ""}",
            currentRound = currentRound,
            totalRounds = totalRounds,
            duration = duration
        )
        
        fun completed() = TrainingProgress(
            phase = TrainingPhase.COMPLETED,
            message = "Training completed successfully!"
        )
        
        fun error(errorMessage: String, errorDetails: String? = null) = TrainingProgress(
            phase = TrainingPhase.ERROR,
            message = "Error: $errorMessage",
            errorDetails = errorDetails
        )
        
        fun stopping() = TrainingProgress(
            phase = TrainingPhase.STOPPING,
            message = "Stopping training..."
        )
    }
}

/**
 * Phases of the federated learning training process.
 * 
 * This enum represents all possible states during a federated learning session,
 * providing a standardized way to track and report training progress.
 */
enum class TrainingPhase {
    IDLE,               // Not training
    CONNECTING,         // Connecting to server
    FETCHING_JOB,       // Fetching job from server
    JOB_RECEIVED,       // Job received from server
    FETCHING_TASK,      // Fetching task from server
    TASK_RECEIVED,      // Task received, ready to train
    TRAINING,           // Actively training
    SENDING_RESULTS,    // Sending results back to server
    RESULTS_SENT,       // Results sent successfully
    COMPLETED,          // Training session completed
    ERROR,              // Error occurred
    STOPPING            // User requested stop
}

