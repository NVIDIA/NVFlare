//
//  TrainerController.swift
//  NVFlareMobile - Demo App
//
//  App-level coordinator that uses NVFlareSDK
//

import Foundation
import UIKit

enum TrainingStatus {
    case idle
    case training
    case stopping
}

enum TrainingError: Error {
    case datasetCreationFailed
    case connectionFailed
    case trainingFailed
}

enum SupportedJob: String, CaseIterable {
    case cifar10 = "CIFAR10"
    case xor = "XOR"
    
    var displayName: String {
        switch self {
        case .cifar10: return "CIFAR-10"
        case .xor: return "XOR"
        }
    }
    
    var datasetType: String {
        switch self {
        case .cifar10: return "cifar10"
        case .xor: return "xor"
        }
    }
}

@MainActor
class TrainerController: ObservableObject {
    @Published var status: TrainingStatus = .idle
    @Published var supportedJobs: Set<SupportedJob> = [.cifar10, .xor]
    
    private var currentTask: Task<Void, Error>?
    private var flareRunner: NVFlareRunner?
    
    // Server configuration
    @Published var serverHost = "192.168.6.101"
    @Published var serverPort = 4321
    
    var capabilities: [String: Any] {
        return [
            "supported_jobs": supportedJobs.map { $0.rawValue }
        ]
    }
    
    func toggleJob(_ job: SupportedJob) {
        if supportedJobs.contains(job) {
            supportedJobs.remove(job)
        } else {
            supportedJobs.insert(job)
        }
        
        // Update the runner if it exists
        flareRunner?.updateSupportedJobs(Array(supportedJobs.map { $0.rawValue }))
    }
    
    func startTraining() async throws {
        guard status == .idle else { return }
        status = .training
        
        currentTask = Task {
            do {
                print("TrainerController: Starting federated learning")
                
                // Create Swift datasets based on supported jobs
                let swiftDataset: SwiftDataset
                if supportedJobs.contains(.cifar10) {
                    swiftDataset = SwiftCIFAR10Dataset()
                    print("TrainerController: Created Swift CIFAR-10 dataset")
                } else if supportedJobs.contains(.xor) {
                    swiftDataset = SwiftXORDataset()
                    print("TrainerController: Created Swift XOR dataset")
                } else {
                    throw TrainingError.datasetCreationFailed
                }
                
                // Create C++ adapter from Swift dataset
                let dataset = SwiftDatasetBridge.createDatasetAdapter(swiftDataset)
                guard dataset != nil else {
                    print("TrainerController: Failed to create C++ dataset adapter!")
                    throw TrainingError.datasetCreationFailed
                }
                
                print("TrainerController: Created C++ dataset adapter from Swift dataset")
                
                // Create FlareRunner with C++ dataset
                let runner = NVFlareRunner(
                    jobName: "federated_learning",  // This will be matched against incoming job names
                    cppDataset: dataset,
                    deviceInfo: [
                        "device_id": UIDevice.current.identifierForVendor?.uuidString ?? "unknown",
                        "platform": "ios",
                        "app_version": Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "unknown"
                    ],
                    userInfo: [:],
                    jobTimeout: 30.0,
                    hostname: serverHost,
                    port: serverPort,
                    supportedJobs: Array(supportedJobs.map { $0.rawValue })
                )
                
                self.flareRunner = runner
                
                print("TrainerController: Supported jobs: \(supportedJobs.map { $0.rawValue })")
                print("TrainerController: Using app's C++ dataset implementation")
                
                // Start the runner (this will call the main FL loop)
                await runner.run()
                
                // Clean up C++ dataset adapter
                SwiftDatasetBridge.destroyDatasetAdapter(dataset)
                
            } catch {
                await MainActor.run {
                    status = .idle
                }
                throw error
            }
        }
        
        try await currentTask?.value
    }
    
    func stopTraining() {
        status = .stopping
        flareRunner?.stop()
        currentTask?.cancel()
        currentTask = nil
        flareRunner = nil
        status = .idle
    }
} 
