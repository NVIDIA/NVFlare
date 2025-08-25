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
    @Published var serverHost = "192.168.6.101"
    @Published var serverPort = 4321
    
    func setJob(_ job: SupportedJob) {
        selectedJob = job
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

                switch selectedJob {
                case .cifar10:
                    do {
                        swiftDataset = try SwiftCIFAR10Dataset()
                                print("TrainerController: Created Swift CIFAR-10 dataset")
        print("TrainerController: CIFAR-10 dataset size: \(swiftDataset.size())")
                        
                        // Reset dataset to ensure we start from the beginning
                        swiftDataset.reset()
                        print("TrainerController: Reset CIFAR-10 dataset for new task")
                    } catch DatasetError.noDataFound {
                        print("TrainerController: CIFAR-10 data not found in app bundle")
                        throw TrainingError.datasetCreationFailed
                    } catch DatasetError.invalidDataFormat {
                        print("TrainerController: CIFAR-10 data format is invalid")
                        throw TrainingError.datasetCreationFailed
                    } catch DatasetError.emptyDataset {
                        print("TrainerController: CIFAR-10 dataset is empty")
                        throw TrainingError.datasetCreationFailed
                    } catch {
                        print("TrainerController: Failed to create CIFAR-10 dataset: \(error)")
                        throw TrainingError.datasetCreationFailed
                    }

                case .xor:
                    swiftDataset = SwiftXORDataset()
                    print("TrainerController: Created Swift XOR dataset")
                    print("TrainerController: XOR dataset size: \(swiftDataset.size())")
                    
                    // Reset dataset to ensure we start from the beginning
                    swiftDataset.reset()
                    print("TrainerController: Reset XOR dataset for new task")
                }

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
                    hostname: serverHost,
                    port: serverPort
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
