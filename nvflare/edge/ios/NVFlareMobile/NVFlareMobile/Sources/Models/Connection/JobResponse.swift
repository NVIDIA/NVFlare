//
//  JobResponse.swift
//  NVFlareMobile
//
//

import Foundation

struct JobResponse: Decodable {
    let status: String
    let jobId: String?
    let jobName: String?
    let jobData: [String: AnyDecodable]?  // Use AnyDecodable wrapper
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

// Common helper type used in many projects
struct AnyDecodable: Decodable {
    let value: Any
    
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self.value = NSNull()
        } else if let bool = try? container.decode(Bool.self) {
            self.value = bool
        } else if let int = try? container.decode(Int.self) {
            self.value = int
        } else if let double = try? container.decode(Double.self) {
            self.value = double
        } else if let string = try? container.decode(String.self) {
            self.value = string
        } else if let array = try? container.decode([AnyDecodable].self) {
            self.value = array.map(\.value)
        } else if let dict = try? container.decode([String: AnyDecodable].self) {
            self.value = dict.mapValues(\.value)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Cannot decode value")
        }
    }
}

extension JobResponse {
    func toJob() throws -> Job {
        guard let jobId = self.jobId else {
            throw NVFlareError.invalidRequest("Can't convert JobResponse to Job")
        }
        
//        // Convert AnyDecodable dictionary to [String: Any]
//        let configDict = Dictionary(uniqueKeysWithValues:
//            jobData.map { (key, value) in (key, value.value) }
//        )
//        
//        // Create JobMeta with defaults if values missing
//        let trainingConfig = TrainingConfig(from: configDict)
        
        return Job(id: jobId, status: "running")
    }
}
