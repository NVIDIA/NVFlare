//
//  ResultResponse.swift
//  NVFlareMobile
//
//

import Foundation

struct ResultResponse: Decodable {
    let status: String
    let taskId: String?
    let taskName: String?
    let jobId: String?
    let message: String?
    let details: [String: String]?
    
    enum CodingKeys: String, CodingKey {
        case status
        case taskId = "task_id"
        case taskName = "task_name"
        case jobId = "job_id"
        case message
        case details
    }
}
