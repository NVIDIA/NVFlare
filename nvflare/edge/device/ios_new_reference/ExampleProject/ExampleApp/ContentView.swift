//
//  ContentView.swift
//  NVFlare ExampleApp
//
//  Example app showing how to use NVFlareSDK
//

import SwiftUI

struct ContentView: View {
    @StateObject private var trainerController = TrainerController()
    
    var body: some View {
        VStack(spacing: 20) {
            Text("NVFlare Mobile Demo")
                .font(.largeTitle)
                .fontWeight(.bold)
            
            // Server Configuration
            VStack(alignment: .leading, spacing: 10) {
                Text("Server Configuration")
                    .font(.headline)
                
                HStack {
                    Text("Host:")
                    TextField("Hostname", text: $trainerController.serverHost)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                }
                
                HStack {
                    Text("Port:")
                    TextField("Port", value: $trainerController.serverPort, formatter: NumberFormatter())
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .keyboardType(.numberPad)
                }
            }
            .padding()
            .background(Color.gray.opacity(0.1))
            .cornerRadius(8)
            
            // Supported Jobs
            VStack(alignment: .leading, spacing: 10) {
                Text("Supported Jobs")
                    .font(.headline)
                
                HStack {
                    JobToggleButton(
                        job: "CIFAR-10",
                        isSupported: trainerController.supportedJobs.contains(.cifar10),
                        action: { trainerController.toggleJob(.cifar10) }
                    )
                    
                    JobToggleButton(
                        job: "XOR", 
                        isSupported: trainerController.supportedJobs.contains(.xor),
                        action: { trainerController.toggleJob(.xor) }
                    )
                }
            }
            .padding()
            .background(Color.gray.opacity(0.1))
            .cornerRadius(8)
            
            // C++ Dataset Status
            VStack(alignment: .leading, spacing: 10) {
                Text("High-Performance C++ Datasets")
                    .font(.headline)
                
                HStack {
                    Image(systemName: "cpu.fill")
                        .foregroundColor(.blue)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("CIFAR10 Dataset")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                        Text("Native C++ • Direct ExecutorTorch • Zero overhead")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    if trainerController.supportedJobs.contains(.cifar10) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                    }
                }
                
                HStack {
                    Image(systemName: "function")
                        .foregroundColor(.purple)
                    VStack(alignment: .leading, spacing: 4) {
                        Text("XOR Dataset")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                        Text("Native C++ • Optimized tensors • Minimal memory")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    if trainerController.supportedJobs.contains(.xor) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                    }
                }
            }
            .padding()
            .background(Color.blue.opacity(0.1))
            .cornerRadius(8)
            
            // Training Status
            VStack(spacing: 10) {
                Text("Status: \(statusText)")
                    .font(.headline)
                    .foregroundColor(statusColor)
                
                Button(action: {
                    Task {
                        if trainerController.status == .idle {
                            try await trainerController.startTraining()
                        } else {
                            trainerController.stopTraining()
                        }
                    }
                }) {
                    Text(trainerController.status == .idle ? "Start Training" : "Stop Training")
                        .foregroundColor(.white)
                        .padding()
                        .background(trainerController.status == .idle ? Color.green : Color.red)
                        .cornerRadius(8)
                }
                .disabled(trainerController.status == .stopping)
            }
            
            Spacer()
        }
        .padding()
    }
    
    private var statusText: String {
        switch trainerController.status {
        case .idle: return "Ready"
        case .training: return "Training"
        case .stopping: return "Stopping..."
        }
    }
    
    private var statusColor: Color {
        switch trainerController.status {
        case .idle: return .blue
        case .training: return .green
        case .stopping: return .orange
        }
    }
}

struct JobToggleButton: View {
    let job: String
    let isSupported: Bool
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(job.uppercased())
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(isSupported ? Color.blue : Color.gray.opacity(0.3))
                .foregroundColor(isSupported ? .white : .gray)
                .cornerRadius(6)
        }
    }
}

#Preview {
    ContentView()
}

