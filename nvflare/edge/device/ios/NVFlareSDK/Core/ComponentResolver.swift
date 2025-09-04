//
//  ComponentResolver.swift
//  NVFlare iOS SDK
//
//  Component resolution system for transforming server configs to iOS native components
//  Similar to Python's ComponentResolver but adapted for iOS/Swift
//

import Foundation

/// Registry that maps server component types to iOS component creators
public class ComponentRegistry {
    private var creators: [String: ComponentCreator.Type] = [:]
    
    /// Singleton instance
    public static let shared = ComponentRegistry()
    
    private init() {
        // Register default components
        registerDefaults()
    }
    
    /// Register a component creator for a specific server type
    public func register(serverType: String, creator: ComponentCreator.Type) {
        creators[serverType] = creator
    }
    
    /// Get creator for a server component type
    public func getCreator(for serverType: String) -> ComponentCreator.Type? {
        return creators[serverType]
    }
    
    /// Get all registered server types (for debugging)
    public func getRegisteredTypes() -> [String] {
        return Array(creators.keys)
    }
    
    /// Register default component mappings
    private func registerDefaults() {
        // Map server types to iOS implementations using specialized resolvers
        register(serverType: "Trainer.DLTrainer", creator: ETTrainerComponentResolver.self)
        // Add more mappings here as needed
        // register(serverType: "Filter.DP", creator: DPFilterExecutor.self)
        // register(serverType: "Optimizer.SGD", creator: SGDOptimizerWrapper.self)
    }
}

/// Resolves a single component specification into an iOS native object
public class ComponentResolver {
    public let componentType: String
    public let componentName: String
    public let componentArgs: [String: Any]
    private let creator: ComponentCreator.Type
    
    public init?(serverType: String, name: String, args: [String: Any]?) {
        guard let creator = ComponentRegistry.shared.getCreator(for: serverType) else {
            print("ComponentResolver: No creator registered for server type '\(serverType)'")
            return nil
        }
        
        self.componentType = serverType
        self.componentName = name
        self.componentArgs = args ?? [:]
        self.creator = creator
    }
    
    /// Resolve the component spec into an iOS native object
    public func resolve() -> Any? {
        return creator.create(name: componentName, args: componentArgs)
    }
}

/// Processes component configuration from server into iOS native objects
public class ConfigProcessor {
    
    /// Process components array from server config
    public static func processComponents(_ components: [[String: Any]]) -> [String: Any] {
        var resolvedObjects: [String: Any] = [:]
        var resolvers: [String: ComponentResolver] = [:]
        
        // Step 1: Create resolvers for each component
        for component in components {
            guard let name = component["name"] as? String,
                  let type = component["type"] as? String else {
                print("ConfigProcessor: Invalid component missing name/type")
                continue
            }
            
            let args = component["args"] as? [String: Any]
            
            guard let resolver = ComponentResolver(serverType: type, name: name, args: args) else {
                print("ConfigProcessor: Failed to create resolver for type '\(type)' name '\(name)'")
                print("ConfigProcessor: Available resolvers: \(ComponentRegistry.shared.getRegisteredTypes())")
                continue
            }
            
            resolvers[name] = resolver
        }
        
        // Step 2: Resolve components (simplified - no dependency resolution for now)
        for (name, resolver) in resolvers {
            if let resolved = resolver.resolve() {
                resolvedObjects[name] = resolved
            }
        }
        
        return resolvedObjects
    }
    
    /// Process full server configuration into iOS format
    public static func processServerConfig(_ serverConfig: [String: Any]) -> [String: Any] {
        var result: [String: Any] = [:]
        
        // Process components
        if let components = serverConfig["components"] as? [[String: Any]] {
            let resolvedComponents = processComponents(components)
            result["components"] = resolvedComponents
        }
        
        // Process executors mapping
        if let executors = serverConfig["executors"] as? [String: String] {
            result["executors"] = executors
        }
        
        return result
    }
}
