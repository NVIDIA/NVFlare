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
        guard let jobData = jobData else {
            // Fallback if no job data
            return [
                "components": [
                    [
                        "type": "ETTrainer",
                        "name": "trainer", 
                        "args": [:]
                    ]
                ]
            ]
        }
        
        // Parse the actual server configuration
        var configData: [String: Any] = [:]
        
        // Extract components from server config
        if case .dictionary(let config) = jobData["config"],
           case .array(let components) = config["components"] {
            
            var parsedComponents: [[String: Any]] = []
            
            for component in components {
                if case .dictionary(let componentDict) = component {
                    var parsedComponent: [String: Any] = [:]
                    
                    // Extract component name
                    if case .string(let name) = componentDict["name"] {
                        parsedComponent["name"] = name
                    }
                    
                    // Extract and map component type (Trainer.DLTrainer -> ETTrainer)
                    if case .string(let type) = componentDict["type"] {
                        parsedComponent["type"] = (type == "Trainer.DLTrainer") ? "ETTrainer" : type
                    }
                    
                    // Extract and convert component args
                    if case .dictionary(let args) = componentDict["args"] {
                        var convertedArgs: [String: Any] = [:]
                        
                        for (key, value) in args {
                            switch value {
                            case .int(let intVal):
                                // Map server parameter names to ETTrainer format
                                if key == "epoch" {
                                    convertedArgs["num_epochs"] = intVal
                                } else {
                                    convertedArgs[key] = intVal
                                }
                            case .double(let doubleVal):
                                // Map server parameter names to ETTrainer format  
                                if key == "lr" {
                                    convertedArgs["learning_rate"] = Float(doubleVal)
                                } else {
                                    convertedArgs[key] = doubleVal
                                }
                            case .string(let stringVal):
                                convertedArgs[key] = stringVal
                            case .bool(let boolVal):
                                convertedArgs[key] = boolVal
                            default:
                                break
                            }
                        }
                        
                        parsedComponent["args"] = convertedArgs
                    }
                    
                    parsedComponents.append(parsedComponent)
                }
            }
            
            configData["components"] = parsedComponents
        }
        
        // Extract executors mapping if present
        if case .dictionary(let config) = jobData["config"],
           case .dictionary(let executors) = config["executors"] {
            
            var parsedExecutors: [String: Any] = [:]
            for (key, value) in executors {
                if case .string(let stringVal) = value {
                    parsedExecutors[key] = stringVal
                }
            }
            configData["executors"] = parsedExecutors
        }
        
        return configData
    }
} 
