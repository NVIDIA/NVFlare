//
//  TrainerController.swift
//  NVFlareMobile - Demo App
//
//  App-level coordinator that uses NVFlareSDK
//

import Foundation
import UIKit

/// Simple data source implementation for ExampleApp
private class SimpleDataSource: NSObject, NVFlareDataSource {
    private let dataset: NVFlareDataset
    
    init(dataset: NVFlareDataset) {
        self.dataset = dataset
    }
    
    func getDataset(datasetType: String, ctx: NVFlareContext) -> NVFlareDataset? {
        return dataset
    }
}

enum TrainingStatus {
    case idle
    case training
    case stopping
}

enum TrainingError: Error {
    case datasetCreationFailed
    case connectionFailed
    case trainingFailed
    case noSupportedJobs
}

enum SupportedJob: String, CaseIterable {
    case cifar10 = "cifar10_et"
    case xor = "xor_et"
    
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
    
    var datasetDescription: String {
        switch self {
        case .cifar10: return "CIFAR10 dataset"
        case .xor: return "Xor toy dataset"
        }
    }
    
    var iconName: String {
        switch self {
        case .cifar10: return "cpu.fill"
        case .xor: return "function"
        }
    }
    
    var iconColor: String {
        switch self {
        case .cifar10: return "blue"
        case .xor: return "purple"
        }
    }
}

@MainActor
class TrainerController: ObservableObject {
    @Published var status: TrainingStatus = .idle
    
    // Only one job can be selected at a time
    @Published var selectedJob: SupportedJob? = .cifar10
    
    private var currentTask: Task<Void, Error>?
    private var flareRunner: NVFlareRunner?
    
    // Prevent dataset from being deallocated during training
    private var currentSwiftDataset: NVFlareDataset?
    
    // Server configuration
    @Published var serverURL = "https://192.168.6.101:443"
    
    func setJob(_ job: SupportedJob) {
        selectedJob = job
    }
    
    private func createDataset(for jobType: SupportedJob) throws -> NVFlareDataset {
        switch jobType {
        case .cifar10:
            do {
                let dataset = try SwiftCIFAR10Dataset()
                print("CIFAR-10 dataset loaded: \(dataset.size()) samples")
                return dataset
            } catch {
                print("CIFAR-10 failed to load: \(error)")
                throw TrainingError.datasetCreationFailed
            }
            
        case .xor:
            do {    
                let dataset = try SwiftXORDataset()
                print("XOR dataset ready: \(dataset.size()) samples")
                return dataset
            } catch {
                print("XOR failed to load: \(error)")
                throw TrainingError.datasetCreationFailed
            }

        }
    }
    
    func startTraining() async throws {
        guard status == .idle else { return }
        guard let selectedJob = selectedJob else {
            throw TrainingError.noSupportedJobs
        }
        
        status = .training
        
        currentTask = Task {
            do {
                print("TrainerController: Starting federated learning")

                let swiftDataset: NVFlareDataset

                // Create dataset based on selected job
                swiftDataset = try createDataset(for: selectedJob)
                swiftDataset.reset()
                print("TrainerController: Dataset ready with \(swiftDataset.size()) samples")

                // Store reference to keep it alive
                self.currentSwiftDataset = swiftDataset
                print("TrainerController: Stored dataset reference: \(swiftDataset)")

                print("TrainerController: Using NVFlareDataset with new SDK")

                let dataSource = SimpleDataSource(dataset: swiftDataset)
                let runner = try NVFlareRunner(
                    jobName: selectedJob.rawValue,
                    dataSource: dataSource,
                    deviceInfo: [
                        "device_id": UIDevice.current.identifierForVendor?.uuidString ?? "unknown",
                        "platform": "ios",
                        "app_version": Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "unknown"
                    ],
                    userInfo: [:],
                    jobTimeout: 30.0,
                    serverURL: serverURL,
                    allowSelfSignedCerts: true
                )
                
                self.flareRunner = runner
                
                print("TrainerController: Created NVFlareRunner with NVFlareDataset")
                print("TrainerController: About to start training with dataset: \(String(describing: self.currentSwiftDataset))")

                await runner.run()
                
                print("TrainerController: Training completed, cleaning up")
                
                await MainActor.run {
                    print("TrainerController: Clearing Swift dataset reference")
                    self.currentSwiftDataset = nil
                    status = .idle
                }

            } catch {
                print("TrainerController: Training failed with error: \(error)")
                await MainActor.run {
                    self.currentSwiftDataset = nil
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
    }
}
