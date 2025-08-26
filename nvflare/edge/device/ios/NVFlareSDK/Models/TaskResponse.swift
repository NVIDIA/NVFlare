//
//  TaskResponse.swift
//  NVFlareMobile
//
//

import Foundation


struct TrainingConfig {
    let totalEpochs: Int
    let batchSize: Int
    let learningRate: Float
    let method: String
    var dataSetType: String  // Derived from method, kept for compatibility
    
    var dictionary: [String: Any] {
        return [
            NVFlareProtocolConstants.metaKeyTotalEpochs: totalEpochs,
            NVFlareProtocolConstants.metaKeyBatchSize: batchSize,
            NVFlareProtocolConstants.metaKeyLearningRate: learningRate,
            "method": method,
            NVFlareProtocolConstants.metaKeyDatasetType: dataSetType
        ]
    }

    // Default initializer
    init() {
        self.totalEpochs = 1
        self.batchSize = 4
        self.learningRate = 0.1
        self.method = TrainingConstants.methodCNN
        self.dataSetType = TrainingConstants.datasetTypeCIFAR10
    }
    
    init(from data: [String: Any]) {
        self.totalEpochs = data[NVFlareProtocolConstants.metaKeyTotalEpochs] as? Int ?? 1
        self.batchSize = data[NVFlareProtocolConstants.metaKeyBatchSize] as? Int ?? 4
        self.learningRate = data[NVFlareProtocolConstants.metaKeyLearningRate] as? Float ?? 0.1
        self.method = data["method"] as? String ?? TrainingConstants.methodCNN
        self.dataSetType = data[NVFlareProtocolConstants.metaKeyDatasetType] as? String ?? TrainingConstants.datasetTypeCIFAR10
    }
}


/// Represents a training task assigned to the device, including its configuration and model data.
/// 
/// `TrainingTask` is used to encapsulate all information required to execute a training job locally,
/// such as the task's unique identifier, job association, model data, and training configuration.
/// 
/// This struct is typically used when a new training task is received and needs to be processed by the device.
/// In contrast, `TaskResponse` is used to represent the response sent back to the server after task execution,
/// including results and status. Use `TrainingTask` for incoming task details, and `TaskResponse` for outgoing results.

struct TrainingTask {
    let id: String
    let name: String
    let jobId: String
    let modelData: String
    let trainingConfig: TrainingConfig
}

enum JSONValue: Codable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case null
    case array([JSONValue])
    case dictionary([String: JSONValue])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        
        if container.decodeNil() {
            self = .null
            return
        }
        
        // Try String
        if let str = try? container.decode(String.self) {
            self = .string(str)
            return
        }
        
        // Try Int
        if let intVal = try? container.decode(Int.self) {
            self = .int(intVal)
            return
        }
        
        // Try Double
        if let doubleVal = try? container.decode(Double.self) {
            self = .double(doubleVal)
            return
        }
        
        // Try Bool
        if let boolVal = try? container.decode(Bool.self) {
            self = .bool(boolVal)
            return
        }
        
        // Try Array
        if let arr = try? container.decode([JSONValue].self) {
            self = .array(arr)
            return
        }
        
        // Try Dictionary
        if let dict = try? container.decode([String: JSONValue].self) {
            self = .dictionary(dict)
            return
        }
        
        // If none matches, throw error
        throw DecodingError.typeMismatch(JSONValue.self, DecodingError.Context(codingPath: decoder.codingPath, debugDescription: "Unsupported JSON value"))
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let str):
            try container.encode(str)
        case .int(let intVal):
            try container.encode(intVal)
        case .double(let doubleVal):
            try container.encode(doubleVal)
        case .bool(let boolVal):
            try container.encode(boolVal)
        case .null:
            try container.encodeNil()
        case .array(let arr):
            try container.encode(arr)
        case .dictionary(let dict):
            try container.encode(dict)
        }
    }
    
    var jsonObject: Any {
        switch self {
        case .string(let str):
            return str
        case .int(let intVal):
            return intVal
        case .double(let doubleVal):
            return doubleVal
        case .bool(let boolVal):
            return boolVal
        case .null:
            return NSNull()
        case .array(let arr):
            return arr.map { $0.jsonObject }
        case .dictionary(let dict):
            return dict.mapValues { $0.jsonObject }
        }
    }
    
    var asDictionary: [String: Any]? {
        switch self {
        case .dictionary(let dict):
            return dict.mapValues { $0.jsonObject }
        default:
            return nil
        }
    }
}

struct TaskResponse: Decodable {
    let status: String
    let message: String?
    let jobId: String?
    let task_id: String?
    let task_name: String?
    let retryWait: Int?
    let task_data: TaskData?
    let cookie: JSONValue?
    
    struct TaskData: Decodable {
        let data: String
        let meta: JSONValue
        let kind: String
    }
    
    enum CodingKeys: String, CodingKey {
        case status
        case message
        case jobId = "job_id"
        case task_id = "task_id"
        case task_name = "task_name"
        case retryWait = "retry_wait"
        case task_data = "task_data"
        case cookie = "cookie"
    }
    
    public enum TaskStatus: String {
        case ok = "OK"
        case done = "DONE"
        case error = "ERROR"
        case retry = "RETRY"
        case unknown
        case noJob = "NO_JOB"
        case noTask = "NO_TASK"
        case invalid = "INVALID"
        
        var isSuccess: Bool {
            switch self {
            case .ok, .done: return true
            case .error, .retry, .unknown, .noJob, .noTask, .invalid: return false
            }
        }
        
        var shouldContinueTraining: Bool {
            return self == .ok
        }
        
        var shouldLookForNewJob: Bool {
            return self == .noJob
        }
        
        var shouldRetryTask: Bool {
            return self == .retry || self == .noTask
        }
        
        var isTerminal: Bool {
            return self == .done || self == .invalid || self == .error
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
        
        // Use server-provided training parameters from task_data.meta if available
        let trainingConfig: TrainingConfig
        if let metaDict = task_data.meta.asDictionary {
            trainingConfig = TrainingConfig(from: metaDict)
        } else {
            trainingConfig = TrainingConfig()
        }
        
        return TrainingTask(id: task_id,
                          name: task_name,
                          jobId: jobId,
                          modelData: task_data.data,
                          trainingConfig: trainingConfig)
    }
}
