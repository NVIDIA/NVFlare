//
//  JobResponse.swift
//  NVFlareMobile
//
//  Created by Yuan-Ting Hsieh on 2/26/25.
//

import Foundation

struct JobResponse: Decodable {
    let status: String
    let sessionId: String?
    let jobId: String?
    let jobName: String?
    let jobData: JobDataResponse?
    let method: String?
    let retryWait: Int?
    let message: String?
    let details: [String: String]?
    
    enum CodingKeys: String, CodingKey {
        case status
        case sessionId = "session_id"
        case jobId = "job_id"
        case jobName = "job_name"
        case jobData = "job_data"
        case method
        case retryWait = "retry_wait"
        case message
        case details
    }
}

// Define the structure of job_data
struct JobDataResponse: Decodable {
    let totalEpochs: Int
    let batchSize: Int
    let learningRate: Float
    
    enum CodingKeys: String, CodingKey {
        case totalEpochs = "total_epochs"
        case batchSize = "batch_size"
        case learningRate = "learning_rate"
    }
}

extension JobResponse {
    func toJob() throws -> (job: Job, sessionId: String?) {
        guard let jobId = self.jobId,
              let jobData = self.jobData else {
            throw NVFlareError.invalidMetadata
        }
        let meta = JobMeta(from: jobData)
        let job = Job(id: jobId, meta: meta, status: "running")
        return (job, self.sessionId)
    }
}

