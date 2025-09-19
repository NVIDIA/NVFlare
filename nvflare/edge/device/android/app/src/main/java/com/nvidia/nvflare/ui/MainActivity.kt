package com.nvidia.nvflare.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.nvidia.nvflare.app.controllers.FlareRunnerController
import com.nvidia.nvflare.sdk.training.TrainingStatus
import com.nvidia.nvflare.app.controllers.SupportedJob
import com.nvidia.nvflare.ui.theme.NVFlareTheme
import kotlinx.coroutines.launch
import java.net.NetworkInterface

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
    var errorMessage by remember { mutableStateOf<String?>(null) }
    var supportedJobsState by remember { mutableStateOf(flareRunnerController.supportedJobs) }
    var useHttpsState by remember { mutableStateOf(flareRunnerController.useHttps) }
    var allowSelfSignedCertsState by remember { mutableStateOf(flareRunnerController.allowSelfSignedCerts) }

    
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
                                if (status == TrainingStatus.TRAINING) {
                                    return@onCheckedChange
                                }
                                
                                flareRunnerController.toggleJob(job)
                                supportedJobsState = flareRunnerController.supportedJobs
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
                                flareRunnerController.startTraining(
                                    onStatusUpdate = { newStatus ->
                                        status = newStatus
                                        errorMessage = null
                                    },
                                    onError = { error ->
                                        status = TrainingStatus.IDLE
                                        errorMessage = error.message ?: "Unknown error"
                                    },
                                    onSuccess = {
                                        status = TrainingStatus.IDLE
                                        errorMessage = null
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
                        },
                        enabled = status == TrainingStatus.TRAINING,
                        modifier = Modifier.weight(1f)
                    ) {
                        Text("Stop Training")
                    }
                }
            }
        }
    }
} 