//
//  NVFlareConfig.swift
//  NVFlare iOS SDK
//
//  Configuration processing
//

import Foundation

// MARK: - Configuration Keys

/// Configuration keys for component definitions
public struct NVFlareConfigKey {
    public static let name = "name"
    public static let type = "type"
    public static let args = "args"
    public static let trainer = "trainer"
    public static let executors = "executors"
    public static let components = "components"
    public static let inFilters = "in_filters"
    public static let outFilters = "out_filters"
    public static let handlers = "handlers"
}

// MARK: - Training Configuration

/// Training configuration result - equivalent to Python TrainConfig
public class NVFlareTrainConfig {
    public let objects: [String: Any]
    public let inFilters: [NVFlareFilter]?
    public let outFilters: [NVFlareFilter]?
    public let eventHandlers: [NVFlareEventHandler]?
    public let executors: [String: NVFlareExecutor]?
    
    public init(objects: [String: Any], 
                inFilters: [NVFlareFilter]?, 
                outFilters: [NVFlareFilter]?, 
                eventHandlers: [NVFlareEventHandler]?, 
                executors: [String: NVFlareExecutor]?) {
        self.objects = objects
        self.inFilters = inFilters
        self.outFilters = outFilters
        self.eventHandlers = eventHandlers
        self.executors = executors
    }
    
    public func findExecutor(taskName: String) -> NVFlareExecutor? {
        if let executors = executors {
            return executors[taskName] ?? executors["*"]
        }
        return objects[NVFlareConfigKey.trainer] as? NVFlareExecutor
    }
}

// MARK: - Configuration Processor

/// Swift-style config processor - equivalent to Python process_train_config
public class NVFlareConfigProcessor {
    
    public static func processTrainConfig(config: [String: Any], 
                                        resolverRegistry: [String: ComponentCreator.Type]) throws -> NVFlareTrainConfig {
        
        print("NVFlareConfigProcessor: Processing config with keys: \(config.keys)")
        print("NVFlareConfigProcessor: Available resolvers: \(resolverRegistry.keys)")
        if let components = config[NVFlareConfigKey.components] {
            print("NVFlareConfigProcessor: Found components of type: \(type(of: components))")
            print("NVFlareConfigProcessor: Components value: \(components)")
        } else {
            print("NVFlareConfigProcessor: No components found in config")
        }
        
        guard let components = config[NVFlareConfigKey.components] as? [[String: Any]] else {
            throw NVFlareConfigError.missingComponents
        }
        
        // Create all components
        var objects: [String: Any] = [:]
        
        for component in components {
            guard let name = component[NVFlareConfigKey.name] as? String,
                  let type = component[NVFlareConfigKey.type] as? String else {
                throw NVFlareConfigError.invalidComponent("Missing name or type")
            }
            
            let args = component[NVFlareConfigKey.args] as? [String: Any] ?? [:]
            
            guard let creatorType = resolverRegistry[type] else {
                throw NVFlareConfigError.noResolver(type)
            }
            
            let object = creatorType.create(name: name, args: args)
            objects[name] = object
        }
        
        // Process input filters
        var inFilters: [NVFlareFilter] = []
        if let inFilterConfigs = config[NVFlareConfigKey.inFilters] as? [[String: Any]] {
            for filterConfig in inFilterConfigs {
                if let filter = try processFilterConfig(filterConfig, resolverRegistry: resolverRegistry) {
                    inFilters.append(filter)
                }
            }
        }
        
        // Process output filters
        var outFilters: [NVFlareFilter] = []
        if let outFilterConfigs = config[NVFlareConfigKey.outFilters] as? [[String: Any]] {
            for filterConfig in outFilterConfigs {
                if let filter = try processFilterConfig(filterConfig, resolverRegistry: resolverRegistry) {
                    outFilters.append(filter)
                }
            }
        }
        
        // Process event handlers
        var eventHandlers: [NVFlareEventHandler] = []
        if let handlerConfigs = config[NVFlareConfigKey.handlers] as? [[String: Any]] {
            for handlerConfig in handlerConfigs {
                if let handler = try processHandlerConfig(handlerConfig, resolverRegistry: resolverRegistry) {
                    eventHandlers.append(handler)
                }
            }
        }
        
        // Process executors
        var executors: [String: NVFlareExecutor] = [:]
        if let executorConfigs = config[NVFlareConfigKey.executors] as? [[String: Any]] {
            for executorConfig in executorConfigs {
                if let (taskName, executor) = try processExecutorConfig(executorConfig, resolverRegistry: resolverRegistry) {
                    executors[taskName] = executor
                }
            }
        }
        
        return NVFlareTrainConfig(
            objects: objects,
            inFilters: inFilters.isEmpty ? nil : inFilters,
            outFilters: outFilters.isEmpty ? nil : outFilters,
            eventHandlers: eventHandlers.isEmpty ? nil : eventHandlers,
            executors: executors.isEmpty ? nil : executors
        )
    }
    
    // MARK: - Helper Methods
    
    private static func processFilterConfig(_ config: [String: Any], 
                                          resolverRegistry: [String: ComponentCreator.Type]) throws -> NVFlareFilter? {
        guard let type = config[NVFlareConfigKey.type] as? String else {
            throw NVFlareConfigError.invalidComponent("Filter missing type")
        }
        
        let args = config[NVFlareConfigKey.args] as? [String: Any] ?? [:]
        
        guard let creatorType = resolverRegistry[type] else {
            throw NVFlareConfigError.noResolver(type)
        }
        
        let object = creatorType.create(name: type, args: args)
        return object as? NVFlareFilter
    }
    
    private static func processHandlerConfig(_ config: [String: Any], 
                                           resolverRegistry: [String: ComponentCreator.Type]) throws -> NVFlareEventHandler? {
        guard let type = config[NVFlareConfigKey.type] as? String else {
            throw NVFlareConfigError.invalidComponent("Handler missing type")
        }
        
        let args = config[NVFlareConfigKey.args] as? [String: Any] ?? [:]
        
        guard let creatorType = resolverRegistry[type] else {
            throw NVFlareConfigError.noResolver(type)
        }
        
        let object = creatorType.create(name: type, args: args)
        return object as? NVFlareEventHandler
    }
    
    private static func processExecutorConfig(_ config: [String: Any], 
                                            resolverRegistry: [String: ComponentCreator.Type]) throws -> (String, NVFlareExecutor)? {
        guard let type = config[NVFlareConfigKey.type] as? String else {
            throw NVFlareConfigError.invalidComponent("Executor missing type")
        }
        
        let taskName = config["task_name"] as? String ?? "*"
        let args = config[NVFlareConfigKey.args] as? [String: Any] ?? [:]
        
        guard let creatorType = resolverRegistry[type] else {
            throw NVFlareConfigError.noResolver(type)
        }
        
        let object = creatorType.create(name: type, args: args)
        if let executor = object as? NVFlareExecutor {
            return (taskName, executor)
        }
        return nil
    }
}

// MARK: - Configuration Errors

/// Configuration errors - equivalent to Python ConfigError
public enum NVFlareConfigError: Error {
    case missingComponents
    case invalidComponent(String)
    case noResolver(String)
    case invalidConfiguration(String)
} 
