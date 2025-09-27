//
//  ResultResponse.swift
//  NVFlare iOS SDK
//
//  Server response model for result reporting
//

import Foundation

struct ResultResponse: Decodable {
    let status: String
    let message: String?
    let taskId: String?
    let taskName: String?
    let retryWait: Int?
    
    enum CodingKeys: String, CodingKey {
        case status
        case message
        case taskId = "task_id"
        case taskName = "task_name"
        case retryWait = "retry_wait"
    }
    
    enum ResultStatus: String {
        case ok = "OK"
        case done = "DONE"
        case error = "ERROR"
        case noJob = "NO_JOB"
        case noTask = "NO_TASK"
        case invalid = "INVALID_REQUEST"
        case unknown
        
        var shouldContinueToTask: Bool {
            return self == .ok || self == .noTask
        }
        
        var shouldGoBackToJob: Bool {
            return self == .noJob
        }
        
        var isTerminal: Bool {
            return self == .done || self == .invalid || self == .error
        }
    }
    
    var resultStatus: ResultStatus {
        return ResultStatus(rawValue: self.status) ?? .unknown
    }
}
