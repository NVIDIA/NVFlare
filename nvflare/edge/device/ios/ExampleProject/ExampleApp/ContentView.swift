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
                    Text("URL:")
                    TextField("Hostname", text: $trainerController.serverURL)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                }

            }
            .padding()
            .background(Color.gray.opacity(0.1))
            .cornerRadius(8)
            
            // Job Selection
            VStack(alignment: .leading, spacing: 10) {
                Text("Select Training Job")
                    .font(.headline)
                
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(SupportedJob.allCases, id: \.rawValue) { job in
                        JobSelectionRow(
                            job: job,
                            isSelected: trainerController.selectedJob == job,
                            action: { trainerController.setJob(job) }
                        )
                    }
                }
            }
            .padding()
            .background(Color.gray.opacity(0.1))
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

struct JobSelectionRow: View {
    let job: SupportedJob
    let isSelected: Bool
    let action: () -> Void
    
    private var iconColor: Color {
        switch job.iconColor {
        case "blue": return .blue
        case "purple": return .purple
        default: return .gray
        }
    }
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                // Radio button
                Image(systemName: isSelected ? "largecircle.fill.circle" : "circle")
                    .foregroundColor(isSelected ? .blue : .gray)
                    .font(.title2)
                
                // Dataset icon
                Image(systemName: job.iconName)
                    .foregroundColor(iconColor)
                    .font(.title2)
                
                // Job info
                VStack(alignment: .leading, spacing: 2) {
                    Text(job.displayName)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.primary)
                    
                    Text(job.datasetDescription)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .lineLimit(1)
                }
                
                Spacer()
                
                if isSelected {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                        .font(.title3)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 12)
            .background(isSelected ? Color.blue.opacity(0.1) : Color.clear)
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(isSelected ? Color.blue : Color.gray.opacity(0.3), lineWidth: 1)
            )
        }
        .buttonStyle(PlainButtonStyle())
    }
}

#Preview {
    ContentView()
}

