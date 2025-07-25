package com.nvidia.nvflare.app.controllers

import android.content.Context
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.nvidia.nvflare.sdk.network.Connection
import com.nvidia.nvflare.sdk.AndroidFlareRunner
import com.nvidia.nvflare.app.data.AndroidDataSource
import com.nvidia.nvflare.sdk.utils.MethodType
import com.nvidia.nvflare.sdk.utils.TrainerType
import com.nvidia.nvflare.sdk.utils.TrainingStatus
import com.nvidia.nvflare.sdk.defs.Filter
import com.nvidia.nvflare.sdk.defs.EventHandler
import com.nvidia.nvflare.sdk.defs.Transform
import com.nvidia.nvflare.sdk.defs.Batch
import com.nvidia.nvflare.sdk.defs.NoOpFilter
import com.nvidia.nvflare.sdk.defs.NoOpEventHandler
import com.nvidia.nvflare.sdk.defs.NoOpTransform
import com.nvidia.nvflare.sdk.defs.SimpleBatch
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Controller that manages FlareRunner instances.
 * Provides the same interface as the old TrainerController but uses the new SDK architecture.
 */
class FlareRunnerController(
    private val context: Context,
    private val connection: Connection
) : ViewModel() {
    private val TAG = "FlareRunnerController"
    
    private val _status = MutableLiveData<TrainingStatus>(TrainingStatus.IDLE)
    val status: LiveData<TrainingStatus> = _status

    private val _trainerType = MutableLiveData<TrainerType>(TrainerType.EXECUTORCH)
    val trainerType: LiveData<TrainerType> = _trainerType

    private val _supportedMethods = MutableLiveData<Set<MethodType>>(setOf(MethodType.CNN, MethodType.XOR))
    val supportedMethods: LiveData<Set<MethodType>> = _supportedMethods

    private var currentFlareRunner: AndroidFlareRunner? = null
    private var currentTask: kotlinx.coroutines.Job? = null

    val capabilities: Map<String, Any>
        get() {
            val methods = _supportedMethods.value?.map { it.displayName } ?: emptyList()
            return mapOf("methods" to methods)
        }

    init {
        // Set initial capabilities
        connection.setCapabilities(capabilities)
    }

    fun toggleMethod(method: MethodType) {
        val currentMethods = _supportedMethods.value ?: emptySet()
        _supportedMethods.value = if (currentMethods.contains(method)) {
            currentMethods - method
        } else {
            currentMethods + method
        }
        connection.setCapabilities(capabilities)
    }

    fun setTrainerType(type: TrainerType) {
        _trainerType.value = type
    }

    fun startTraining() {
        if (_status.value == TrainingStatus.TRAINING) {
            Log.w(TAG, "Training already in progress")
            return
        }

        _status.value = TrainingStatus.TRAINING
        currentTask = viewModelScope.launch {
            try {
                runTrainingWithFlareRunner()
            } catch (e: Exception) {
                Log.e(TAG, "Training failed", e)
                if (_status.value != TrainingStatus.STOPPING) {
                    _status.value = TrainingStatus.IDLE
                }
                throw e
            }
        }
    }

    fun stopTraining() {
        _status.value = TrainingStatus.STOPPING
        currentFlareRunner?.stop()
        currentTask?.cancel()
        currentTask = null
        currentFlareRunner = null
        _status.value = TrainingStatus.IDLE
        connection.resetCookie()
    }

    private suspend fun runTrainingWithFlareRunner() = withContext(Dispatchers.IO) {
        try {
            Log.d(TAG, "Creating FlareRunner")
            
            // Create device info
            val deviceInfo = createDeviceInfo()
            
            // Create user info
            val userInfo = createUserInfo()
            
            // Create data source
            val dataSource = AndroidDataSource()
            
            // Create resolver registry for components
            val resolverRegistry = createResolverRegistry()
            
            // Create FlareRunner
            currentFlareRunner = AndroidFlareRunner(
                context = context,
                connection = connection,
                dataSource = dataSource,
                deviceInfo = deviceInfo,
                userInfo = userInfo,
                jobTimeout = 300.0f, // 5 minutes timeout
                resolverRegistry = resolverRegistry
            )
            
            Log.d(TAG, "Starting FlareRunner")
            currentFlareRunner?.run()
            
        } catch (e: Exception) {
            Log.e(TAG, "FlareRunner execution failed", e)
            throw e
        }
    }
    
    private fun createDeviceInfo(): Map<String, String> {
        return mapOf(
            "device_id" to if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.P) {
                context.packageManager.getPackageInfo(context.packageName, 0).longVersionCode.toString()
            } else {
                context.packageManager.getPackageInfo(context.packageName, 0).versionCode.toString()
            },
            "app_name" to "test",
            "app_version" to context.packageManager.getPackageInfo(context.packageName, 0).versionName,
            "platform" to "android",
            "platform_version" to "1.2.2"
        )
    }
    
    private fun createUserInfo(): Map<String, String> {
        return mapOf(
            "user_id" to "xyz"
        )
    }
    
    private fun createResolverRegistry(): Map<String, Class<*>> {
        return mapOf(
            "Executor.AndroidExecutor" to AndroidExecutor::class.java,
            "DataSource.AndroidDataSource" to AndroidDataSource::class.java,
            "Filter.NoOpFilter" to NoOpFilter::class.java,
            "EventHandler.NoOpEventHandler" to NoOpEventHandler::class.java,
            "Transform.NoOpTransform" to NoOpTransform::class.java,
            "Batch.SimpleBatch" to SimpleBatch::class.java
        )
    }
} 