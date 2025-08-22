//
//  NVFlareRunner.swift
//  NVFlare iOS SDK
//
//  iOS FlareRunner implementation - the main entry point for iOS federated learning
//

import Foundation
import UIKit

/// iOS FlareRunner - main entry point for iOS federated learning
public class NVFlareRunner: ObservableObject {
    
    // MARK: - Public Properties
    @Published public private(set) var status: NVFlareStatus = .idle
    
    // MARK: - Configuration Properties  
    public var jobName: String
    private let dataSource: NVFlareDataSource        // Standard data source protocol
    private let dataset: NVFlareDataset              // Current dataset instance
    private let cppDataset: UnsafeMutableRawPointer?  // Internal C++ ETDataset* (optional for safety)
    public let deviceInfo: [String: String]
    public let userInfo: [String: String]
    public let jobTimeout: TimeInterval
    public let abortSignal = NVFlareSignal()
    
    // MARK: - Filter Properties
    public let appInFilters: [NVFlareFilter]?
    public let appOutFilters: [NVFlareFilter]?
    
    // MARK: - Internal State
    public var resolverRegistry: [String: ComponentCreator.Type] = [:]
    public var jobId: String?
    public var cookie: [String: Any]?
    
    // iOS-specific properties
    private let connection: NVFlareConnection
    
    /// Primary initializer - using standard NVFlare interfaces
    public init(
        jobName: String,
        dataSource: NVFlareDataSource,
        deviceInfo: [String: String],
        userInfo: [String: String],
        jobTimeout: TimeInterval,
        hostname: String = "",
        port: Int = 0,
        inFilters: [NVFlareFilter]? = nil,
        outFilters: [NVFlareFilter]? = nil,
        resolverRegistry: [String: ComponentCreator.Type]? = nil
    ) throws {
        self.jobName = jobName
        self.dataSource = dataSource
        
        // Get dataset from data source based on job name
        let ctx = NVFlareContext()
        guard let dataset = dataSource.getDataset(datasetType: jobName, ctx: ctx) else {
            print("NVFlareRunner: Failed to get dataset for job: \(jobName)")
            throw DatasetError.noDataFound
        }
        
        self.dataset = dataset
        print("NVFlareRunner: Got dataset from data source for job: \(jobName)")
        print("NVFlareRunner: Dataset size: \(dataset.size())")
        
        // Convert NVFlareDataset to C++ using bridge
        guard let cppDatasetPtr = SwiftDatasetBridge.createDatasetAdapter(dataset) else {
            var errorMessage = "NVFlareRunner: Failed to create C++ dataset adapter!"
            // Attempt to get more info from SwiftDatasetBridge if available
            if let bridgeType = SwiftDatasetBridge.self as? AnyObject,
               let lastError = (bridgeType.value(forKey: "lastErrorMessage") as? String), !lastError.isEmpty {
                errorMessage += " Reason: \(lastError)"
            } else {
                errorMessage += " Dataset type: \(type(of: dataset)), size: \(dataset.size())"
            }
            print(errorMessage)
            throw DatasetError.dataLoadFailed
        }
        
        self.cppDataset = cppDatasetPtr
        print("NVFlareRunner: Created C++ dataset adapter from NVFlareDataset")
        
        self.deviceInfo = deviceInfo
        self.userInfo = userInfo
        self.jobTimeout = jobTimeout
        self.appInFilters = inFilters
        self.appOutFilters = outFilters
        self.connection = NVFlareConnection(hostname: hostname, port: port, deviceInfo: deviceInfo)
        
        // Add built-in resolvers
        addBuiltinResolvers()
        
        // Add app-provided resolvers (can override built-in ones)
        if let resolverRegistry = resolverRegistry {
            self.resolverRegistry.merge(resolverRegistry) { _, new in new }
        }
        
    }
    
    deinit {
        if let cppDataset = cppDataset {
            SwiftDatasetBridge.destroyDatasetAdapter(cppDataset)
            print("NVFlareRunner: Cleaned up C++ dataset adapter")
        } else {
            print("NVFlareRunner: No C++ dataset adapter to clean up (initialization may have failed)")
        }
    }
    
    // MARK: - Main Run Loop
    
    /// Main run loop - equivalent to Python FlareRunner.run()
    public func run() async {
        while true {
            let sessionDone = await doOneJob()
            if sessionDone {
                return
            }
        }
    }
    
    /// Stop the runner
    public func stop() {
        abortSignal.trigger(true)
    }

    
    // MARK: - iOS-Specific Implementation
    
    /// Add ExecuTorch-specific resolvers  
    private func addBuiltinResolvers() {
        resolverRegistry.merge([
            "ETTrainer": ETTrainerExecutor.self,
            "Trainer.DLTrainer": ETTrainerExecutor.self,  // Map server trainer type to ETTrainer
        ]) { _, new in new }
    }
    
    
    /// iOS implementation of job fetching
    private func getJob(ctx: NVFlareContext, abortSignal: NVFlareSignal) async -> NVFlareJob? {
        while !abortSignal.triggered {
            do {
                let jobResponse = try await connection.fetchJob(jobName: self.jobName)
                
                // Check for terminal states - no job for me
                if jobResponse.jobStatus.isTerminalState {
                    return nil
                }
                
                // Check if we have a valid job to return
                if jobResponse.jobStatus.hasValidJob {
                    return jobResponse.toNVFlareJob()
                }
                
                // For retry/error/unknown status, wait and continue loop
                let retryWait = jobResponse.retryWait ?? 5
                try await Task.sleep(nanoseconds: UInt64(retryWait) * 1_000_000_000)
                
            } catch {
                print("Failed to get job: \(error), retrying...")
                try? await Task.sleep(nanoseconds: 5_000_000_000)
                continue
            }
        }
        
        return nil
    }
    
    /// iOS implementation of task fetching
    private func getTask(ctx: NVFlareContext, abortSignal: NVFlareSignal) async -> TaskResult {
        guard let jobId = self.jobId else {
            return TaskResult.sessionDone()
        }
        
        while !abortSignal.triggered {
            do {
                let taskResponse = try await connection.fetchTask(jobId: jobId)
                
                if taskResponse.taskStatus == .done {
                    return TaskResult.sessionDone()
                }
                
                if taskResponse.taskStatus.shouldContinueTraining {
                    if let taskId = taskResponse.task_id,
                       let taskName = taskResponse.task_name,
                       let taskData = taskResponse.task_data {
                        
                        let task = NVFlareTask(
                            taskId: taskId,
                            taskName: taskName,
                            taskData: [
                                "kind": taskData.kind,
                                "data": ["model": taskData.data],
                                "meta": taskData.meta.jsonObject
                            ],
                            cookie: taskResponse.cookie?.jsonObject as? [String: Any] ?? [:]
                        )
                        
                        return TaskResult.continuing(with: task)
                    }
                }
                
                // Wait before retrying
                let retryWait = taskResponse.retryWait ?? 5
                try await Task.sleep(nanoseconds: UInt64(retryWait) * 1_000_000_000)
                
            } catch {
                print("Failed to get task: \(error), retrying...")
                try? await Task.sleep(nanoseconds: 5_000_000_000)
                continue
            }
        }
        
        return TaskResult.sessionDone()
    }
    
    /// iOS implementation of result reporting
    private func reportResult(result: [String: Any], ctx: NVFlareContext, abortSignal: NVFlareSignal) async -> Bool {
        guard let jobId = self.jobId,
              let taskId = ctx[NVFlareContextKey.taskId] as? String,
              let taskName = ctx[NVFlareContextKey.taskName] as? String else {
            return true
        }
        
        do {
            try await connection.sendResult(jobId: jobId, taskId: taskId, taskName: taskName, weightDiff: result)
            return false // Continue training
        } catch {
            print("Failed to report result: \(error)")
            return true // Stop session on error
        }
    }
    
    // MARK: - Filter Processing
    
    /// Apply filters to DXO data - equivalent to Python FlareRunner._do_filtering()
    private func doFiltering(data: NVFlareDXO, filters: [NVFlareFilter]?, ctx: NVFlareContext) -> NVFlareDXO {
        guard let filters = filters, !filters.isEmpty else {
            return data
        }
        
        var filteredData = data
        for filter in filters {
            filteredData = filter.filter(data: filteredData, ctx: ctx, abortSignal: abortSignal)
            if abortSignal.triggered {
                break
            }
        }
        return filteredData
    }
    
    // MARK: - FL Implementation (Same as Python)
    
    /// Process one job - equivalent to Python FlareRunner._do_one_job()
    public func doOneJob() async -> Bool {
        let ctx = NVFlareContext()
        ctx[NVFlareContextKey.runner] = self
        
        print("NVFlareRunner: Storing dataset in context (NVFlareDataset converted to C++)")
        print("NVFlareRunner: NVFlareDataset size: \(dataset.size())")
        
        guard let cppDataset = cppDataset else {
            print("NVFlareRunner: ERROR - No C++ dataset available (initialization failed)")
            return true  // Cannot continue without dataset
        }
        
        ctx[NVFlareContextKey.dataset] = cppDataset
        
        // Try to get job
        guard let job = await getJob(ctx: ctx, abortSignal: abortSignal) else {
            return true // No job for me
        }
        
        guard !job.jobName.isEmpty else {
            print("NVFlareRunner: Missing or empty job_name in job data")
            return false
        }
        guard !job.jobId.isEmpty else {
            print("NVFlareRunner: Missing or empty job_id in job data")
            return false
        }
        
        self.jobName = job.jobName
        self.jobId = job.jobId
        let configData = job.configData
        
        // Process training configuration
        do {
            let trainConfig = try NVFlareConfigProcessor.processTrainConfig(
                config: configData, 
                resolverRegistry: resolverRegistry
            )
            ctx[NVFlareContextKey.components] = trainConfig.objects
            ctx[NVFlareContextKey.eventHandlers] = trainConfig.eventHandlers
            
            // Set up filters - app filters first, then job filters
            var inFilters: [NVFlareFilter] = []
            if let appInFilters = self.appInFilters {
                inFilters.append(contentsOf: appInFilters)
            }
            if let jobInFilters = trainConfig.inFilters {
                inFilters.append(contentsOf: jobInFilters)
            }
            
            var outFilters: [NVFlareFilter] = []
            if let appOutFilters = self.appOutFilters {
                outFilters.append(contentsOf: appOutFilters)
            }
            if let jobOutFilters = trainConfig.outFilters {
                outFilters.append(contentsOf: jobOutFilters)
            }
            
            // Task execution loop
            while !abortSignal.triggered {
                let taskResult = await getTask(ctx: ctx, abortSignal: abortSignal)
                
                if abortSignal.triggered { return true }
                guard let task = taskResult.task else { return taskResult.sessionDone }
                
                // Create new context for each task
                let taskCtx = NVFlareContext()
                let enumerator = ctx.keyEnumerator()
                while let key = enumerator.nextObject() {
                    if let value = ctx.object(forKey: key),
                       let copyableKey = key as? NSCopying {
                        taskCtx.setObject(value, forKey: copyableKey)
                    }
                }
                
                // Ensure dataset is available in task context
                taskCtx[NVFlareContextKey.dataset] = cppDataset
                
                self.cookie = task.cookie
                let taskName = task.taskName
                let taskData = task.taskData
                
                // Convert task data to DXO (should work now with correct "kind" key)
                guard let taskDXO = NVFlareDXO.fromDict(taskData) else {
                    print("NVFlareRunner: FATAL ERROR - Failed to create DXO from task data: \(taskData)")
                    print("NVFlareRunner: This indicates a serious protocol mismatch. Stopping training.")
                    return true  // Stop the session
                }
                
                // Find the right executor
                guard let executor = trainConfig.findExecutor(taskName: taskName) else {
                    print("Cannot find executor for task: \(taskName)")
                    continue
                }
                
                taskCtx[NVFlareContextKey.taskId] = task.taskId
                taskCtx[NVFlareContextKey.taskName] = taskName
                taskCtx[NVFlareContextKey.taskData] = taskData
                taskCtx[NVFlareContextKey.executor] = executor
                
                if abortSignal.triggered { return true }
                
                // Filter the input
                let filteredInput = doFiltering(data: taskDXO, filters: inFilters.isEmpty ? nil : inFilters, ctx: taskCtx)
                
                if abortSignal.triggered { return true }
                
                // Fire before train event
                taskCtx.fireEvent(
                    eventType: NVFlareEventType.beforeTrain, 
                    data: Date().timeIntervalSince1970, 
                    abortSignal: abortSignal
                )
                
                // Execute training
                let output = executor.execute(taskData: filteredInput, ctx: taskCtx, abortSignal: abortSignal)
                
                // Fire after train event
                taskCtx.fireEvent(
                    eventType: NVFlareEventType.afterTrain, 
                    data: (Date().timeIntervalSince1970, output), 
                    abortSignal: abortSignal
                )
                
                if abortSignal.triggered { return true }
                
                // Filter the output
                let filteredOutput = doFiltering(data: output, filters: outFilters.isEmpty ? nil : outFilters, ctx: taskCtx)
                
                if abortSignal.triggered { return true }
                
                // Report result
                let sessionDoneAfterReport = await reportResult(result: filteredOutput.toDict(), ctx: taskCtx, abortSignal: abortSignal)
                if sessionDoneAfterReport { return sessionDoneAfterReport }
                if abortSignal.triggered { return true }
            }
            return true
            
        } catch {
            print("Failed to process job configuration: \(error)")
            return true
        }
    }

}

/// Public status enum
public enum NVFlareStatus {
    case idle
    case training
    case stopping
}
