//
//  TaskResponse.swift
//  NVFlareMobile
//
//  Created by Yuan-Ting Hsieh on 2/26/25.
//

import Foundation

struct TaskResponse: Decodable {
    let status: String
    let taskId: String?
    let taskName: String?
    let taskData: [String: String]?
    let retryWait: Int?
    let message: String?
    let details: [String: String]?
    
    enum CodingKeys: String, CodingKey {
        case status
        case taskId = "task_id"
        case taskName = "task_name"
        case taskData = "task_data"
        case retryWait = "retry_wait"
        case message
        case details
    }
}

extension TaskResponse {
    enum TaskStatus: String {
        case ok = "OK"
        case finished = "FINISHED"
        case retry = "RETRY"
        case unknown
        
        init(rawValue: String) {
            switch rawValue {
            case "OK": self = .ok
            case "FINISHED": self = .finished // TODO:: change to DONE
            case "RETRY": self = .retry
            default: self = .unknown
            }
        }
        
        var shouldContinueTraining: Bool {
            switch self {
            case .ok: return true
            case .finished: return false
            case .retry: return true
            case .unknown: return false
            }
        }
    }
    
    var taskStatus: TaskStatus {
        return TaskStatus(rawValue: self.status)
    }

    func toTrainingTask(jobId: String) throws -> TrainingTask {
        guard let taskId = self.taskId,
              let taskData = self.taskData else {
            throw NVFlareError.taskFetchFailed
        }
        return TrainingTask(id: taskId, jobId: jobId, modelData: taskData)
    }
}
