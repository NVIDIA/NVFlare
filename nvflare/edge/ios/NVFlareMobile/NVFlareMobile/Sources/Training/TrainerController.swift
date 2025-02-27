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

enum TrainerType {
    case executorch
    case coreML
}


@MainActor
class TrainerController: ObservableObject {
    @Published var status: TrainingStatus = .idle
    @Published var trainerType: TrainerType = .executorch
    private var currentTask: Task<Void, Error>?
    
    private let connection: Connection
    private let deviceStateMonitor: DeviceStateMonitor
    
    init(connection: Connection) {
        self.connection = connection
        self.deviceStateMonitor = DeviceStateMonitor()
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
    }
    
    private func runTrainingLoop() async throws {
        // Job fetching loop
        var currentJob: Job?
        var sessionId: String?
        
        while currentJob == nil && !Task.isCancelled {
            do {
                let jobResponse = try await connection.fetchJob()
                if jobResponse.status == "stopped" {
                    throw NVFlareError.serverRequestedStop
                }
                
                let (job, sid) = try jobResponse.toJob()
                currentJob = job
                sessionId = sid
                
            } catch {
                print("Failed to fetch job \(error), retrying in 5 seconds...")
                try await Task.sleep(nanoseconds: 5 * 1_000_000_000)
                continue
            }
        }
        
        guard let job = currentJob, let sid = sessionId else {
            throw NVFlareError.jobFetchFailed
        }
        
        // Task execution loop
        while job.status == "running" && !Task.isCancelled {
            do {
                let taskResponse = try await connection.fetchTask(sessionId: sid, jobId: job.id)

                if !taskResponse.taskStatus.shouldContinueTraining {
                    print("Training finished - no more tasks")
                    return
                }

                let task = try taskResponse.toTrainingTask(jobId: job.id)
                
                let trainer = try createTrainer(withModelData: task.modelData, meta: job.meta)
                
                // Check device state before heavy computation
                guard deviceStateMonitor.isReadyForTraining else {
                    throw NVFlareError.trainingFailed("Device not ready")
                }
                
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
                    sessionId: sid,
                    taskId: task.id,
                    taskName: "train",
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
    
    private func createTrainer(withModelData modelData: [String: String], meta: JobMeta) throws -> Trainer {
        switch trainerType {
        case .executorch:
            guard let modelString = modelData["model"] else {
                // Handle missing or invalid model data
                throw NVFlareError.invalidModelData
            }
            return ETTrainerWrapper(modelBase64: modelString, meta: meta)
        case .coreML:
            fatalError("CoreML trainer not implemented yet")
        }
    }

}
