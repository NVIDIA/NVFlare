//
//  NVFlareTypes.swift
//  NVFlare iOS SDK
//
//  Core types and protocols for the NVFlare SDK
//

import Foundation

// MARK: - Public SDK Types

/// Signal class for handling abort conditions
public class NVFlareSignal: NSObject {
    public private(set) var triggered = false
    
    public func trigger(_ value: Bool) {
        triggered = value
    }
    
    public func reset() {
        triggered = false
    }
}

/// Context keys for storing/retrieving objects from Context
public struct NVFlareContextKey {
    public static let runner = "runner"
    public static let dataset = "dataset"  // C++ ETDataset* pointer (app's dataset)
    public static let executor = "executor"
    public static let components = "components"
    public static let eventHandlers = "event_handlers"
    public static let taskName = "task_name"
    public static let taskId = "task_id"
    public static let taskData = "task_data"
}

/// Event types for training lifecycle events
public struct NVFlareEventType {
    public static let beforeTrain = "before_train"
    public static let afterTrain = "after_train"
    public static let lossGenerated = "loss_generated"
}

/// Context dictionary for passing data between components
public class NVFlareContext: NSMutableDictionary {
    private var storage = NSMutableDictionary()
    
    // MARK: - Required NSMutableDictionary Methods
    
    public override var count: Int {
        return storage.count
    }
    
    public override func object(forKey aKey: Any) -> Any? {
        return storage.object(forKey: aKey)
    }
    
    public override func setObject(_ anObject: Any, forKey aKey: NSCopying) {
        storage.setObject(anObject, forKey: aKey)
    }
    
    public override func removeObject(forKey aKey: Any) {
        storage.removeObject(forKey: aKey)
    }
    
    public override func keyEnumerator() -> NSEnumerator {
        return storage.keyEnumerator()
    }
    
    // MARK: - NVFlare Methods
    
    public func fireEvent(eventType: String, data: Any, abortSignal: NVFlareSignal) {
        guard let handlers = self[NVFlareContextKey.eventHandlers] as? [NVFlareEventHandler] else { return }
        
        for handler in handlers {
            handler.handleEvent(eventType: eventType, eventData: data, ctx: self, abortSignal: abortSignal)
        }
    }
    
    public subscript(key: String) -> Any? {
        get { return object(forKey: key) }
        set {
            if let value = newValue {
                setObject(value, forKey: key as NSString)
            } else {
                removeObject(forKey: key)
            }
        }
    }
}

/// DXO (Data Exchange Object)
public class NVFlareDXO: NSObject {
    public let dataKind: String
    public let data: [String: Any]
    public let meta: [String: Any]
    
    public init(dataKind: String, data: [String: Any], meta: [String: Any] = [:]) {
        self.dataKind = dataKind
        self.data = data
        self.meta = meta
        super.init()
    }
    
    public func toDict() -> [String: Any] {
        return [
            "kind": dataKind,
            "data": data,
            "meta": meta
        ]
    }
    
    public static func fromDict(_ dict: [String: Any]) -> NVFlareDXO? {
        guard let dataKind = dict["kind"] as? String,
              let data = dict["data"] as? [String: Any] else {
            return nil
        }
        let meta = dict["meta"] as? [String: Any] ?? [:]
        return NVFlareDXO(dataKind: dataKind, data: data, meta: meta)
    }
}

// MARK: - Public SDK Protocols

/// Batch protocol for training data batches
@objc public protocol NVFlareBatch {
    @objc(getInput) func getInput() -> Any
    @objc(getLabel) func getLabel() -> Any
    /// Optional: batch size for verification
    @objc(getBatchSize) func getBatchSize() -> Int
}

/// Concrete implementation of NVFlareBatch
public class NVFlareDataBatch: NSObject, NVFlareBatch {
    private let input: Any
    private let label: Any
    private let batchSize: Int
    
    public init(input: Any, label: Any, batchSize: Int) {
        self.input = input
        self.label = label
        self.batchSize = batchSize
        super.init()
    }
    
    @objc(getInput) public func getInput() -> Any {
        return input
    }
    
    @objc(getLabel) public func getLabel() -> Any {
        return label
    }
    
    @objc(getBatchSize) public func getBatchSize() -> Int {
        return batchSize
    }
}

/// Dataset protocol for training datasets  
@objc public protocol NVFlareDataset {
    func size() -> Int
    @objc(getNextBatchWithBatchSize:) func getNextBatch(batchSize: Int) -> NVFlareBatch
    func reset()
    /// Optional: get input dimensions (width, height, channels, etc.)
    @objc(getInputDimensions) optional func getInputDimensions() -> [Int]
    /// Optional: get number of classes/output dimensions  
    @objc(getOutputDimensions) optional func getOutputDimensions() -> [Int]
}

/// DataSource protocol for providing datasets
@objc public protocol NVFlareDataSource {
    func getDataset(datasetType: String, ctx: NVFlareContext) -> NVFlareDataset?
}

/// Executor protocol for executing training tasks
@objc public protocol NVFlareExecutor {
    func execute(taskData: NVFlareDXO, ctx: NVFlareContext, abortSignal: NVFlareSignal) -> NVFlareDXO
}

/// Filter protocol for filtering input/output data
@objc public protocol NVFlareFilter {
    func filter(data: NVFlareDXO, ctx: NVFlareContext, abortSignal: NVFlareSignal) -> NVFlareDXO
}

/// Transform protocol for transforming batches during training
@objc public protocol NVFlareTransform {
    func transform(batch: NVFlareBatch, ctx: NVFlareContext, abortSignal: NVFlareSignal) -> NVFlareBatch
}

/// EventHandler protocol for handling training events
@objc public protocol NVFlareEventHandler {
    func handleEvent(eventType: String, eventData: Any, ctx: NVFlareContext, abortSignal: NVFlareSignal)
}

/// Component creator protocol for creating components from configuration
@objc public protocol ComponentCreator {
    static func create(name: String, args: [String: Any]) -> Any
}

/// Job data structure with type safety
public struct NVFlareJob {
    public let jobId: String
    public let jobName: String
    public let configData: [String: Any]
    
    public init(jobId: String, jobName: String, configData: [String: Any]) {
        self.jobId = jobId
        self.jobName = jobName
        self.configData = configData
    }
}

/// Task data structure with type safety
public struct NVFlareTask {
    public let taskId: String
    public let taskName: String
    public let taskData: [String: Any]
    public let cookie: [String: Any]
    
    public init(taskId: String, taskName: String, taskData: [String: Any], cookie: [String: Any]) {
        self.taskId = taskId
        self.taskName = taskName
        self.taskData = taskData
        self.cookie = cookie
    }
}

/// Result from reportResult that indicates next FSM state
public enum ReportResultState {
    case continueTask        // OK/NO_TASK -> continue to getTask
    case lookForNewJob      // NO_JOB -> go back to getJob  
    case sessionDone        // DONE/INVALID/ERROR -> END
}

/// Task result that includes both task data and session status
public struct TaskResult {
    public let task: NVFlareTask?
    public let sessionDone: Bool
    public let jobCompleted: Bool
    
    public init(task: NVFlareTask?, sessionDone: Bool, jobCompleted: Bool = false) {
        self.task = task
        self.sessionDone = sessionDone
        self.jobCompleted = jobCompleted
    }
    
    /// Convenience initializer for session done with no task
    public static func sessionDone() -> TaskResult {
        return TaskResult(task: nil, sessionDone: true, jobCompleted: false)
    }
    
    /// Convenience initializer for job completed (look for new jobs)
    public static func jobCompleted() -> TaskResult {
        return TaskResult(task: nil, sessionDone: false, jobCompleted: true)
    }
    
    /// Convenience initializer for continuing with a task
    public static func continuing(with task: NVFlareTask) -> TaskResult {
        return TaskResult(task: task, sessionDone: false, jobCompleted: false)
    }
}


