package com.nvidia.nvflare.app.utils

import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.os.Environment
import android.os.StatFs
import android.os.SystemClock
import android.os.storage.StorageManager
import java.io.File

/**
 * Monitors device state conditions to determine if training should proceed.
 * Moved to app layer as it's application-specific functionality.
 */
class DeviceStateMonitor(private val context: Context) {
    private val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    private val storageManager = context.getSystemService(Context.STORAGE_SERVICE) as StorageManager
    
    val isReadyForTraining: Boolean
        get() = true  // For now, always return true. We can add actual device state checks later: hasSufficientBattery() && hasSufficientStorage() &&  hasSufficientMemory() && !isDeviceOverheating()

    private fun hasSufficientBattery(): Boolean {
        val batteryLevel = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        } else {
            // For older devices, we'll assume battery is sufficient
            100
        }
        
        // Don't train if battery is below 20%
        return batteryLevel >= 20
    }

    private fun hasSufficientStorage(): Boolean {
        val storageDir = context.getExternalFilesDir(null) ?: return false
        val stat = StatFs(storageDir.path)
        val availableBytes = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR2) {
            stat.availableBytes
        } else {
            stat.availableBlocks.toLong() * stat.blockSize.toLong()
        }
        
        // Require at least 500MB free space
        return availableBytes >= 500 * 1024 * 1024
    }

    private fun hasSufficientMemory(): Boolean {
        val runtime = Runtime.getRuntime()
        val usedMemory = runtime.totalMemory() - runtime.freeMemory()
        val maxMemory = runtime.maxMemory()
        
        // Don't train if more than 80% of memory is used
        return usedMemory < maxMemory * 0.8
    }

    private fun isDeviceOverheating(): Boolean {
        // Check if device has been running for too long without a break
        val uptimeMillis = SystemClock.uptimeMillis()
        
        // If device has been running for more than 4 hours, consider it overheating
        return uptimeMillis > 4 * 60 * 60 * 1000
    }
}
