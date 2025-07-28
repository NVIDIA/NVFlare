//
//  JobResponse.swift
//  NVFlare iOS SDK
//
//  Server response models for job and task communication
//

import Foundation

struct JobResponse: Decodable {
    let status: String
    let jobId: String?
    let jobName: String?
    let jobData: [String: JSONValue]?
    let method: String?
    let retryWait: Int?
    let message: String?
    let details: [String: String]?
    
    enum CodingKeys: String, CodingKey {
        case status
        case jobId = "job_id"
        case jobName = "job_name"
        case jobData = "job_data"
        case method
        case retryWait = "retry_wait"
        case message
        case details
    }
}



extension JobResponse {
    /// Convert JobResponse to domain job model
    func toNVFlareJob() -> NVFlareJob? {
        guard let jobId = self.jobId,
              let jobName = self.jobName else {
            return nil
        }
        
        return NVFlareJob(
            jobId: jobId,
            jobName: jobName,
            configData: toConfigData(from: jobData)
        )
    }
    
    /// Convert JobResponse to data suitable for ConfigProcessor
    func toConfigData(from jobData: [String: JSONValue]?) -> [String: Any] {
        // TODO: get real device side config, use jobData parameter when ready
        return [
            "components": [
                [
                    "type": "ETTrainer",
                    "name": "trainer", 
                    "args": [:]
                ]
            ]
        ] // Testing fallback
    }
} 
