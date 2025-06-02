//
//  TrainerController.swift
//  NVFlareMobile
//
//

import Foundation

enum TrainingStatus {
    case idle
    case training
    case stopping
}

enum TrainerType: String, CaseIterable {
    case executorch = "ExecutorTorch"
    // Add more trainer types as needed
}

enum MethodType: String, CaseIterable {
    case cnn = "cnn"
    case xor = "xor"
    
    var displayName: String {
        switch self {
        case .cnn: return "cnn"
        case .xor: return "xor"
        }
    }
    
    // The dataset type required for this method
    var requiredDataset: String {
        switch self {
        case .cnn: return "cifar10" // NSString * const kDatasetTypeCIFAR10 = @"cifar10";
        case .xor: return "xor" // NSString * const kDatasetTypeXOR = @"xor";
        }
    }
}

@MainActor
class TrainerController: ObservableObject {
    @Published var status: TrainingStatus = .idle
    @Published var trainerType: TrainerType = .executorch
    @Published var supportedMethods: Set<MethodType> = [.cnn, .xor]  // Track supported methods
    
    private var currentTask: Task<Void, Error>?
    
    private let connection: Connection
    private let deviceStateMonitor: DeviceStateMonitor
    
    var capabilities: [String: Any] {
        return [
            "methods": supportedMethods.map { $0.rawValue }
        ]
    }
    
    init(connection: Connection) {
        self.connection = connection
        self.deviceStateMonitor = DeviceStateMonitor()
        // Set initial capabilities
        connection.setCapabilities(capabilities)
    }
    
    func toggleMethod(_ method: MethodType) {
        if supportedMethods.contains(method) {
            supportedMethods.remove(method)
        } else {
            supportedMethods.insert(method)
        }
        // Update capabilities when supported methods change
        connection.setCapabilities(capabilities)
    }
    
    func startTraining() async throws {
        guard status == .idle else { return }
        status = .training
        
        currentTask = Task {
            do {
                try await runTrainingLoop()
            } catch {
                status = .idle
                throw error
            }
        }
        
        try await currentTask?.value
    }
    
    func stopTraining() {
        status = .stopping
        currentTask?.cancel()
        currentTask = nil
        status = .idle
        connection.resetCookie()
    }
    
    private func runTrainingLoop() async throws {
        var currentJob: Job?
        
        while currentJob == nil && !Task.isCancelled {
            do {
                let jobResponse = try await connection.fetchJob()
                if jobResponse.status == "stopped" {
                    throw NVFlareError.serverRequestedStop
                }
                
                let job = try jobResponse.toJob()
                // Verify that we support this job's method
                let methodString = jobResponse.method ?? ""  // Use empty string as fallback
                if let method = MethodType(rawValue: methodString),
                   supportedMethods.contains(method) {
                    currentJob = job
                } else {
                    print("Skipping job with unsupported or missing method: \(methodString)")
                    continue  // Skip this job and try to fetch another one
                }
                
            } catch {
                print("Failed to fetch job \(error), retrying in 5 seconds...")
                try await Task.sleep(nanoseconds: 5 * 1_000_000_000)
                continue
            }
        }
        
        guard let job = currentJob else {
            throw NVFlareError.jobFetchFailed
        }
        
        // Task execution loop
        while job.status == "running" && !Task.isCancelled {
            do {
                let taskResponse = try await connection.fetchTask(jobId: job.id)

                if !taskResponse.taskStatus.shouldContinueTraining {
                    print("Training finished - no more tasks")
                    return
                }

                print("task response:    \(taskResponse)")
                let task = try taskResponse.toTrainingTask(jobId: job.id)
                
                let trainer = try createTrainer(withModelData: task.modelData, meta: task.trainingConfig)
                
//                // Check device state before heavy computation
//                guard deviceStateMonitor.isReadyForTraining else {
//                    throw NVFlareError.trainingFailed("Device not ready")
//                }
                
                print("before calling trainer.train")
                
                // Train and get weight differences
                let weightDiff = try await Task.detached(priority: .background) {
                    do {
                        return try await trainer.train()
                    } catch {
                        print("Training failed: \(error)")
                        throw error
                    }
                }.value
                
                // Check device state again before sending results
                guard deviceStateMonitor.isReadyForTraining else {
                    throw NVFlareError.trainingFailed("Device no longer ready")
                }
                
                // Send results back
                try await connection.sendResult(
                    jobId: job.id,
                    taskId: task.id,
                    taskName: task.name,
                    weightDiff: weightDiff
                )
                
            } catch {
                print("Task execution failed: \(error)")
                if status != .stopping {
                    status = .idle
                }
                throw error
            }
        }
    }
    
    private func createTrainer(withModelData modelData: String, meta: TrainingConfig) throws -> Trainer {
        // Get the method from the job metadata
        let methodString = meta.method ?? ""  // Use empty string as fallback
        guard let method = MethodType(rawValue: methodString) else {
            throw NVFlareError.invalidMetadata("Missing or invalid method in job metadata")
        }
        
        // Verify that we support this method
        guard supportedMethods.contains(method) else {
            throw NVFlareError.invalidMetadata("Method \(methodString) is not supported by this client")
        }

        switch trainerType {
        case .executorch:
            print("before calling ETTrainerWrapper")
            return ETTrainerWrapper(modelBase64: modelData, meta: meta)
        default:
            fatalError("trainer not implemented yet")
        }
    }

}
