package com.nvidia.nvflare.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.nvidia.nvflare.app.controllers.FlareRunnerController
import com.nvidia.nvflare.sdk.models.TrainingProgress
import com.nvidia.nvflare.sdk.models.TrainingPhase
import com.nvidia.nvflare.sdk.training.TrainingStatus
import com.nvidia.nvflare.app.controllers.SupportedJob
import com.nvidia.nvflare.ui.theme.NVFlareTheme
import kotlinx.coroutines.launch
import java.net.NetworkInterface
import java.text.SimpleDateFormat
import java.util.ArrayDeque
import java.util.Date
import java.util.Locale

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            NVFlareTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen()
                }
            }
        }
    }
}

@Composable
fun MainScreen() {
    val context = LocalContext.current
    val flareRunnerController = remember { FlareRunnerController(context) }
    val scope = rememberCoroutineScope()
    
    var hostnameText by remember { mutableStateOf(flareRunnerController.serverHost) }
    var portText by remember { mutableStateOf(flareRunnerController.serverPort.toString()) }
    var status by remember { mutableStateOf(TrainingStatus.IDLE) }
    var trainingProgress by remember { mutableStateOf(TrainingProgress.idle()) }
    var progressHistory by remember { mutableStateOf(ArrayDeque<TrainingProgress>(100)) }
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var supportedJobsState by remember { mutableStateOf(flareRunnerController.supportedJobs) }
    var useHttpsState by remember { mutableStateOf(flareRunnerController.useHttps) }
    var allowSelfSignedCertsState by remember { mutableStateOf(flareRunnerController.allowSelfSignedCerts) }
    var sessionStartTime by remember { mutableStateOf<Long?>(null) }
    
    // UI State
    var showDebugDetails by remember { mutableStateOf(false) }
    var showActivityLog by remember { mutableStateOf(true) }

    
    // Get IP address
    val ipAddress = remember {
        try {
            NetworkInterface.getNetworkInterfaces()
                .asSequence()
                .flatMap { it.inetAddresses.asSequence() }
                .filter { !it.isLoopbackAddress && it.hostAddress.indexOf(':') < 0 }
                .firstOrNull()?.hostAddress ?: "No IP found"
        } catch (e: Exception) {
            "Error getting IP"
        }
    }
    
    // Update controller when text changes
    LaunchedEffect(hostnameText) {
        flareRunnerController.serverHost = hostnameText
    }
    LaunchedEffect(portText) {
        flareRunnerController.serverPort = portText.toIntOrNull() ?: 4321
    }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // IP Address Display
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = "Device IP Address",
                    style = MaterialTheme.typography.titleMedium
                )
                Text(
                    text = ipAddress,
                    style = MaterialTheme.typography.bodyLarge
                )
            }
        }

        // Server Configuration
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "Server Configuration",
                    style = MaterialTheme.typography.titleMedium
                )
                
                OutlinedTextField(
                    value = hostnameText,
                    onValueChange = { hostnameText = it },
                    label = { Text("Hostname") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )
                
                OutlinedTextField(
                    value = portText,
                    onValueChange = { portText = it },
                    label = { Text("Port") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number)
                )
                
                // SSL Configuration
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Use HTTPS",
                        modifier = Modifier.weight(1f)
                    )
                    Switch(
                        checked = useHttpsState,
                        onCheckedChange = { checked ->
                            useHttpsState = checked
                            flareRunnerController.useHttps = checked
                        }
                    )
                }
                
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Allow Self-Signed Certs",
                        modifier = Modifier.weight(1f)
                    )
                    Switch(
                        checked = allowSelfSignedCertsState,
                        onCheckedChange = { checked ->
                            allowSelfSignedCertsState = checked
                            flareRunnerController.allowSelfSignedCerts = checked
                        }
                    )
                }
            }
        }

        // Supported Jobs
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "Supported Jobs",
                    style = MaterialTheme.typography.titleMedium
                )
                

                
                SupportedJob.values().forEach { job ->
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = job.displayName,
                            modifier = Modifier.weight(1f)
                        )
                        Switch(
                            checked = supportedJobsState.contains(job),
                            onCheckedChange = { checked ->
                                // Prevent changes during training to avoid race conditions
                                if (status != TrainingStatus.TRAINING) {
                                    flareRunnerController.toggleJob(job)
                                    supportedJobsState = flareRunnerController.supportedJobs
                                }
                            },
                            enabled = status != TrainingStatus.TRAINING,
                            modifier = Modifier.padding(start = 8.dp)
                        )
                    }
                }
            }
        }

        // Training Control
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "Training Control",
                    style = MaterialTheme.typography.titleMedium
                )
                
                // Status Display
                Text(
                    text = "Status: ${status.name}",
                    style = MaterialTheme.typography.bodyMedium
                )
                
                // Error Display
                errorMessage?.let { error ->
                    Text(
                        text = "Error: $error",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.error
                    )
                }
                
                // Control Buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Button(
                        onClick = {
                            scope.launch {
                                // Reset progress history and start tracking
                                progressHistory = ArrayDeque(100)
                                sessionStartTime = System.currentTimeMillis()
                                
                                flareRunnerController.startTraining(
                                    onStatusUpdate = { newStatus ->
                                        status = newStatus
                                        errorMessage = null
                                    },
                                    onProgressUpdate = { progress ->
                                        trainingProgress = progress
                                        // Add to history with efficient ArrayDeque operations
                                        val updatedHistory = progressHistory
                                        if (updatedHistory.size >= 100) {
                                            updatedHistory.removeFirst()
                                        }
                                        updatedHistory.addLast(progress)
                                        progressHistory = updatedHistory
                                    },
                                    onError = { error ->
                                        status = TrainingStatus.IDLE
                                        errorMessage = error.message ?: "Unknown error"
                                        trainingProgress = TrainingProgress.error(
                                            error.message ?: "Unknown error",
                                            error.stackTraceToString()
                                        )
                                    },
                                    onSuccess = {
                                        status = TrainingStatus.IDLE
                                        errorMessage = null
                                        trainingProgress = TrainingProgress.completed()
                                    }
                                )
                            }
                        },
                        enabled = status == TrainingStatus.IDLE,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Start Training")
                    }
                    
                    Button(
                        onClick = {
                            flareRunnerController.stopTraining()
                            status = TrainingStatus.IDLE
                            errorMessage = null
                            trainingProgress = TrainingProgress.stopping()
                        },
                        enabled = status == TrainingStatus.TRAINING,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Stop Training")
                    }
                }
            }
        }

        // Training Progress Display
        if (status == TrainingStatus.TRAINING || trainingProgress.phase != TrainingPhase.IDLE) {
            // Main Status Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = when (trainingProgress.phase) {
                        TrainingPhase.ERROR -> MaterialTheme.colorScheme.errorContainer
                        TrainingPhase.COMPLETED -> MaterialTheme.colorScheme.primaryContainer
                        TrainingPhase.TRAINING -> MaterialTheme.colorScheme.secondaryContainer
                        else -> MaterialTheme.colorScheme.surfaceVariant
                    }
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    // Header with phase
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = trainingProgress.phase.name.replace("_", " "),
                            style = MaterialTheme.typography.titleLarge,
                            color = when (trainingProgress.phase) {
                                TrainingPhase.ERROR -> MaterialTheme.colorScheme.error
                                TrainingPhase.COMPLETED -> MaterialTheme.colorScheme.primary
                                TrainingPhase.TRAINING -> MaterialTheme.colorScheme.secondary
                                else -> MaterialTheme.colorScheme.onSurfaceVariant
                            }
                        )
                        
                        // Session duration
                        sessionStartTime?.let { startTime ->
                            val duration = (System.currentTimeMillis() - startTime) / 1000
                            Text(
                                text = "${duration}s",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                    
                    Divider()
                    
                    // Current Message
                    Text(
                        text = trainingProgress.message,
                        style = MaterialTheme.typography.bodyLarge
                    )
                    
                    // Round Progress Bar
                    if (trainingProgress.currentRound != null && trainingProgress.totalRounds != null && trainingProgress.totalRounds!! > 0) {
                        Column(
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Text(
                                    text = "Round Progress",
                                    style = MaterialTheme.typography.labelLarge
                                )
                                Text(
                                    text = "${trainingProgress.currentRound}/${trainingProgress.totalRounds}",
                                    style = MaterialTheme.typography.bodyMedium
                                )
                            }
                            LinearProgressIndicator(
                                progress = trainingProgress.currentRound!!.toFloat() / trainingProgress.totalRounds!!.toFloat(),
                                modifier = Modifier.fillMaxWidth().height(8.dp)
                            )
                        }
                    }
                    
                    // Quick Stats Grid
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        trainingProgress.datasetSize?.let { size ->
                            StatChip("Dataset", "$size samples")
                        }
                        trainingProgress.duration?.let { dur ->
                            StatChip("Duration", "${dur}ms")
                        }
                    }
                    
                    // Error Details (if present)
                    trainingProgress.errorDetails?.let { details ->
                        if (details.isNotEmpty()) {
                            OutlinedCard(
                                modifier = Modifier.fillMaxWidth(),
                                colors = CardDefaults.cardColors(
                                    containerColor = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.3f)
                                )
                            ) {
                                Column(
                                    modifier = Modifier.padding(12.dp),
                                    verticalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Text(
                                        text = "Error Details",
                                        style = MaterialTheme.typography.labelMedium,
                                        color = MaterialTheme.colorScheme.error
                                    )
                                    Text(
                                        text = details.take(200) + if (details.length > 200) "..." else "",
                                        style = MaterialTheme.typography.bodySmall,
                                        fontFamily = FontFamily.Monospace,
                                        color = MaterialTheme.colorScheme.onErrorContainer
                                    )
                                }
                            }
                        }
                    }
                }
            }
            
            // Debug Details (Collapsible)
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Debug Details",
                            style = MaterialTheme.typography.titleMedium
                        )
                        IconButton(onClick = { showDebugDetails = !showDebugDetails }) {
                            Icon(
                                imageVector = if (showDebugDetails) Icons.Filled.KeyboardArrowUp else Icons.Filled.KeyboardArrowDown,
                                contentDescription = if (showDebugDetails) "Collapse" else "Expand"
                            )
                        }
                    }
                    
                    if (showDebugDetails) {
                        Divider()
                        
                        // Server Info
                        trainingProgress.serverUrl?.let { url ->
                            DebugRow("Server", url)
                        }
                        
                        // Job Info
                        trainingProgress.jobId?.let { jobId ->
                            if (jobId.isNotEmpty()) {
                                DebugRow("Job ID", jobId)
                            }
                        }
                        trainingProgress.jobName?.let { jobName ->
                            if (jobName.isNotEmpty()) {
                                DebugRow("Job Name", jobName)
                            }
                        }
                        
                        // Task Info
                        trainingProgress.taskName?.let { taskName ->
                            if (taskName.isNotEmpty()) {
                                DebugRow("Task", taskName)
                            }
                        }
                        
                        // Dataset Info
                        trainingProgress.datasetSize?.let { size ->
                            DebugRow("Dataset Size", "$size samples")
                        }
                        
                        // Timestamps
                        val dateFormat = SimpleDateFormat("HH:mm:ss.SSS", Locale.getDefault())
                        DebugRow("Last Update", dateFormat.format(Date(trainingProgress.timestamp)))
                    }
                }
            }
            
            // Activity Log (Collapsible, Scrollable)
            if (progressHistory.isNotEmpty()) {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Column(
                        modifier = Modifier.padding(16.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = "Activity Log (${progressHistory.size})",
                                style = MaterialTheme.typography.titleMedium
                            )
                            IconButton(onClick = { showActivityLog = !showActivityLog }) {
                                Icon(
                                    imageVector = if (showActivityLog) Icons.Filled.KeyboardArrowUp else Icons.Filled.KeyboardArrowDown,
                                    contentDescription = if (showActivityLog) "Collapse" else "Expand"
                                )
                            }
                        }
                        
                        if (showActivityLog) {
                            Divider()
                            
                            val dateFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
                            
                            // Show last 20 entries, most recent first
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .heightIn(max = 300.dp)
                                    .verticalScroll(rememberScrollState()),
                                verticalArrangement = Arrangement.spacedBy(4.dp)
                            ) {
                                // Compute the list once before the loop
                                val recentHistory = progressHistory.toList().takeLast(20).reversed()
                                recentHistory.forEachIndexed { index, progress ->
                                    Row(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(vertical = 4.dp),
                                        horizontalArrangement = Arrangement.SpaceBetween
                                    ) {
                                        Column(
                                            modifier = Modifier.weight(1f)
                                        ) {
                                            Text(
                                                text = progress.message,
                                                style = MaterialTheme.typography.bodySmall
                                            )
                                            Text(
                                                text = "${progress.phase.name} • ${dateFormat.format(Date(progress.timestamp))}",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                                            )
                                        }
                                        
                                        // Show round if available
                                        if (progress.currentRound != null && progress.totalRounds != null) {
                                            Text(
                                                text = "${progress.currentRound}/${progress.totalRounds}",
                                                style = MaterialTheme.typography.bodySmall,
                                                modifier = Modifier.padding(start = 8.dp)
                                            )
                                        }
                                    }
                                    // Show divider for all items except the last one
                                    if (index < recentHistory.size - 1) {
                                        Divider(modifier = Modifier.padding(vertical = 2.dp))
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun StatChip(label: String, value: String) {
    Surface(
        shape = MaterialTheme.shapes.small,
        color = MaterialTheme.colorScheme.surface,
        modifier = Modifier.padding(2.dp)
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
            )
            Text(
                text = value,
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}

@Composable
fun DebugRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            text = "$label:",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
            modifier = Modifier.weight(0.4f)
        )
        Text(
            text = value,
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace,
            modifier = Modifier.weight(0.6f)
        )
    }
} 