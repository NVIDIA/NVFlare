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
    
    public enum JobStatus: String {
        case ok = "OK"
        case stopped = "stopped"
        case done = "DONE"
        case running = "RUNNING"
        case submitted = "SUBMITTED"
        case approved = "APPROVED"
        case dispatched = "DISPATCHED"
        case error = "ERROR"
        case retry = "RETRY"
        case unknown
        
        var isTerminalState: Bool {
            switch self {
            case .stopped, .done:
                return true
            case .ok, .running, .submitted, .approved, .dispatched, .error, .retry, .unknown:
                return false
            }
        }
        
        var hasValidJob: Bool {
            switch self {
            case .ok, .running, .submitted, .approved, .dispatched:
                return true
            case .stopped, .done, .error, .retry, .unknown:
                return false
            }
        }
        
        var shouldRetry: Bool {
            switch self {
            case .retry, .unknown:
                return true
            case .ok, .stopped, .done, .running, .submitted, .approved, .dispatched, .error:
                return false
            }
        }
    }
    
    var jobStatus: JobStatus {
        return JobStatus(rawValue: self.status) ?? .unknown
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
    /// Now uses ComponentResolver system instead of hardcoded mappings
    func toConfigData(from jobData: [String: JSONValue]?) -> [String: Any] {
        guard let jobData = jobData else {
            // Fallback if no job data - create default ETTrainer config
            return [
                "components": [
                    [
                        "type": "Trainer.DLTrainer",  // Use server type, let resolver handle mapping
                        "name": "trainer", 
                        "args": [:]
                    ]
                ]
            ]
        }
        
        // Parse server configuration and convert JSONValue to standard types
        guard case .dictionary(let config) = jobData["config"] else {
            print("JobResponse: No config found in job data")
            return [:]
        }
        
        let serverConfig = convertJSONValueToStandardTypes(config)
        
        // Return the server config directly - let NVFlareConfigProcessor handle component resolution
        return serverConfig
    }
    
    /// Convert JSONValue dictionary to standard Swift types for ConfigProcessor
    private func convertJSONValueToStandardTypes(_ jsonDict: [String: JSONValue]) -> [String: Any] {
        var result: [String: Any] = [:]
        
        for (key, jsonValue) in jsonDict {
            result[key] = convertJSONValue(jsonValue)
        }
        
        return result
    }
    
    /// Recursively convert JSONValue to standard Swift types
    private func convertJSONValue(_ jsonValue: JSONValue) -> Any {
        switch jsonValue {
        case .int(let intVal):
            return intVal
        case .double(let doubleVal):
            return doubleVal
        case .string(let stringVal):
            return stringVal
        case .bool(let boolVal):
            return boolVal
        case .array(let arrayVal):
            return arrayVal.map { convertJSONValue($0) }
        case .dictionary(let dictVal):
            return convertJSONValueToStandardTypes(dictVal)
        case .null:
            return NSNull()
        }
    }
} 
