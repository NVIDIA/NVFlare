//
//  TaskResponse.swift
//  NVFlareMobile
//
//

import Foundation

struct TaskResponse: Decodable {
    let status: String
    let message: String?
    let jobId: String?
    let task_id: String?
    let task_name: String?
    let retryWait: Int?
    let task_data: TaskData?
    
    struct TaskData: Decodable {
        let payload: [String: String]
        let task_id: String
    }
    
    enum CodingKeys: String, CodingKey {
        case status
        case message
        case jobId = "job_id"
        case task_id = "task_id"
        case task_name = "task_name"
        case retryWait = "retry_wait"
        case task_data = "task_data"
    }
    
    enum TaskStatus: String {
        case ok = "OK"
        case error = "ERROR"
        case retry = "retry"
        case unknown
        
        var isSuccess: Bool {
            switch self {
            case .ok: return true
            case .error, .retry, .unknown: return false
            }
        }
        
        var shouldContinueTraining: Bool {
            return self == .ok || self == .error
        }

    }
    
    var taskStatus: TaskStatus {
        return TaskStatus(rawValue: self.status) ?? .unknown
    }
    
    func toTrainingTask(jobId: String) throws -> TrainingTask {
        guard taskStatus.shouldContinueTraining else {
            throw NVFlareError.taskFetchFailed(message ?? "Task status indicates training should not continue")
        }
        
        guard let task_id = self.task_id,
              let task_name = self.task_name,
              let task_data = self.task_data else {
            throw NVFlareError.taskFetchFailed("Missing required task data")
        }
        
        return TrainingTask(id: task_id,
                          name: task_name,
                          jobId: jobId,
                          modelData: task_data.payload)
    }
}
