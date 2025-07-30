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
import com.nvidia.nvflare.sdk.network.Connection
import com.nvidia.nvflare.training.MethodType
import com.nvidia.nvflare.sdk.FlareRunnerController
import com.nvidia.nvflare.training.TrainingStatus
import com.nvidia.nvflare.training.TrainerType
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
    val connection = remember { Connection(context) }
    val flareRunnerController = remember { FlareRunnerController(context, connection) }
    val scope = rememberCoroutineScope()
    
    var hostnameText by remember { mutableStateOf(connection.hostname.value ?: "") }
    var portText by remember { mutableStateOf(connection.port.value?.toString() ?: "") }
    
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
    
    // Update connection when text changes
    LaunchedEffect(hostnameText) {
        connection.hostname.value = hostnameText
    }
    LaunchedEffect(portText) {
        connection.port.value = portText.toIntOrNull() ?: 0
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
        
        // Hostname and Port Input
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
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            modifier = Modifier.fillMaxWidth(),
            singleLine = true
        )
        
        // Trainer Type Selection
        Text(
            text = "Trainer Type",
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(top = 8.dp)
        )
        
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            TrainerType.values().forEach { type ->
                FilterChip(
                    selected = flareRunnerController.trainerType.value == type,
                    onClick = { flareRunnerController.setTrainerType(type) },
                    label = { Text(type.name) },
                    modifier = Modifier.weight(1f)
                )
            }
        }
        
        // Supported Methods Section
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "Supported Methods",
                    style = MaterialTheme.typography.titleMedium
                )
                
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 200.dp)
                        .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    MethodType.values().forEach { method ->
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(method.displayName)
                            Switch(
                                checked = flareRunnerController.supportedMethods.value?.contains(method) ?: false,
                                onCheckedChange = { flareRunnerController.toggleMethod(method) }
                            )
                        }
                    }
                }
            }
        }
        
        // Training Button
        Button(
            onClick = {
                if (flareRunnerController.status.value == TrainingStatus.TRAINING) {
                    flareRunnerController.stopTraining()
                } else {
                    scope.launch {
                        flareRunnerController.startTraining()
                    }
                }
            },
            modifier = Modifier.fillMaxWidth(),
            enabled = flareRunnerController.status.value != TrainingStatus.STOPPING &&
                     (flareRunnerController.supportedMethods.value?.isNotEmpty() ?: false) &&
                     connection.isValid
        ) {
            Text(
                if (flareRunnerController.status.value == TrainingStatus.TRAINING) "Stop Training" else "Start Training"
            )
        }
        
        // Progress Indicator
        if (flareRunnerController.status.value == TrainingStatus.TRAINING) {
            CircularProgressIndicator(
                modifier = Modifier.align(Alignment.CenterHorizontally)
            )
        }
    }
} 