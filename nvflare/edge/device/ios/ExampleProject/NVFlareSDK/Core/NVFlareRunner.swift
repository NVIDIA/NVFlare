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
    
    // MARK: - Configuration Properties (using C++ datasets)
    public var jobName: String
    public let cppDataset: UnsafeMutableRawPointer  // C++ ETDataset*
    public let deviceInfo: [String: String]
    public let userInfo: [String: String]
    public let jobTimeout: TimeInterval
    public let abortSignal = NVFlareSignal()
    
    // MARK: - Filter Properties
    public let appInFilters: [NVFlareFilter]?
    public let appOutFilters: [NVFlareFilter]?
    
    // MARK: - Job Support (for backward compatibility)
    public var supportedJobs: [String] = []
    
    // MARK: - Internal State
    public var resolverRegistry: [String: ComponentCreator.Type] = [:]
    public var jobId: String?
    public var cookie: [String: Any]?
    
    // iOS-specific properties
    private let connection: NVFlareConnection
    
    /// Primary initializer - using C++ datasets
    public init(
        jobName: String,
        cppDataset: UnsafeMutableRawPointer,  // App's C++ ETDataset*
        deviceInfo: [String: String],
        userInfo: [String: String],
        jobTimeout: TimeInterval,
        hostname: String = "",
        port: Int = 0,
        supportedJobs: [String] = [],
        inFilters: [NVFlareFilter]? = nil,
        outFilters: [NVFlareFilter]? = nil,
        resolverRegistry: [String: ComponentCreator.Type]? = nil
    ) {
        self.jobName = jobName
        self.cppDataset = cppDataset
        self.deviceInfo = deviceInfo
        self.userInfo = userInfo
        self.jobTimeout = jobTimeout
        self.supportedJobs = supportedJobs
        self.appInFilters = inFilters
        self.appOutFilters = outFilters
        self.connection = NVFlareConnection(hostname: hostname, port: port, deviceInfo: deviceInfo)
        
        // Add built-in resolvers
        addBuiltinResolvers()
        
        // Add app-provided resolvers (can override built-in ones)
        if let resolverRegistry = resolverRegistry {
            self.resolverRegistry.merge(resolverRegistry) { _, new in new }
        }
        
        // Set up capabilities for backward compatibility with server
        setupCapabilities()
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
    
    /// Update supported jobs and refresh capabilities
    public func updateSupportedJobs(_ jobs: [String]) {
        supportedJobs = jobs
        setupCapabilities()
    }
    
    // MARK: - iOS-Specific Implementation
    
    /// Add ExecutorTorch-specific resolvers  
    private func addBuiltinResolvers() {
        resolverRegistry.merge([
            "ETTrainer": ETTrainerExecutor.self,
        ]) { _, new in new }
    }
    
    /// Set up capabilities for backward compatibility with server
    private func setupCapabilities() {
        // Map job names to method names for backward compatibility
        let methods = supportedJobs.compactMap { jobName -> String? in
            switch jobName {
            case "CIFAR10":
                return "cnn"
            case "XOR":
                return "xor"
            default:
                return jobName.lowercased() // fallback
            }
        }
        
        let capabilities: [String: Any] = [
            "methods": methods,                    // Legacy support
            "supported_jobs": supportedJobs,      // New approach
            "device_info": deviceInfo
        ]
        
        connection.setCapabilities(capabilities)
        print("NVFlareRunner: Set capabilities - methods: \(methods), jobs: \(supportedJobs)")
    }
    
    /// iOS implementation of job fetching
    private func getJob(ctx: NVFlareContext, abortSignal: NVFlareSignal) async -> NVFlareJob? {
        while !abortSignal.triggered {
            do {
                let jobResponse = try await connection.fetchJob()
                
                if jobResponse.status == "stopped" || jobResponse.status == "DONE" {
                    return nil
                }
                
                return jobResponse.toNVFlareJob()
                
                // Wait before retrying
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
                
                if taskResponse.status == "DONE" {
                    return TaskResult.sessionDone()
                }
                
                if taskResponse.status == "OK" {
                    if let taskId = taskResponse.task_id,
                       let taskName = taskResponse.task_name,
                       let taskData = taskResponse.task_data {
                        
                        let task = NVFlareTask(
                            taskId: taskId,
                            taskName: taskName,
                            taskData: [
                                "data_kind": taskData.kind,
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
        
        print("🔍 NVFlareRunner: Storing dataset in context, pointer: \(String(describing: cppDataset))")
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
                // Copy all data from original context EXCEPT the dataset pointer
                let enumerator = ctx.keyEnumerator()
                while let key = enumerator.nextObject() {
                    if let value = ctx.object(forKey: key),
                       let copyableKey = key as? NSCopying {
                        // Don't copy the dataset pointer - it gets deleted by ETTrainer
                        if key as? String != NVFlareContextKey.dataset {
                            taskCtx.setObject(value, forKey: copyableKey)
                        }
                    }
                }
                
                // Create a NEW dataset for each task since ETTrainer takes ownership
                if let originalDataset = ctx[NVFlareContextKey.dataset] as? UnsafeMutableRawPointer {
                    print("🔍 NVFlareRunner: Creating new dataset for task (original was: \(originalDataset))")
                    // Create a fresh dataset pointer for this task
                    let newDataset = CreateAppXORDataset()
                    if let newDataset = newDataset {
                        print("🔍 NVFlareRunner: New dataset created: \(newDataset)")
                        taskCtx[NVFlareContextKey.dataset] = newDataset
                    } else {
                        print("❌ NVFlareRunner: Failed to create new dataset for task")
                    }
                }
                
                self.cookie = task.cookie
                let taskName = task.taskName
                let taskData = task.taskData
                let taskDXO = NVFlareDXO.fromDict(taskData)!
                
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
