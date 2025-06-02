//
//  Connection.swift
//  NVFlareMobile
//
//

import Foundation
import UIKit


public class Connection: ObservableObject {
    @Published public var hostname: String = ""
    @Published public var port: Int = 0
    
    private let deviceId = UIDevice.current.identifierForVendor?.uuidString ?? "unknown"
    
    // Minimal device info to reduce privacy concerns
    private var deviceInfo: [String: String] {
        return [
            "device_id": deviceId,
            "platform": "ios",  // Just platform identifier
            "app_version": Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "unknown"  // App version
        ]
    }

    private let jobEndpoint = "job"
    private let taskEndpoint = "task"
    private let resultEndpoint = "result"
    private let scheme = "http"
    private var capabilities: [String: Any] = ["methods": []]
    private var currentCookie: JSONValue?

    public var isValid: Bool {
        return !hostname.isEmpty && port > 0 && port <= 65535
    }
    
    public var serverURL: String {
        return "http://\(hostname):\(port)"
    }
    
    public init(hostname: String = "", port: Int = 0) {
        self.hostname = hostname
        self.port = port
    }
    
    func setCapabilities(_ capabilities: [String: Any]) {
        self.capabilities = capabilities
    }
    
    public func resetCookie() {
        currentCookie = nil
    }
    
    private func getURL(for endpoint: String) -> URL? {
        return URL(string: serverURL)?.appendingPathComponent(endpoint)
    }
    
    public func infoToQueryString(info: [String: String]) -> String {
        return info.map { key, value in
            "\(key)=\(value)"
        }.joined(separator: "&")
    }
    
    func fetchJob() async throws -> JobResponse {
        guard let url = URL(string: "\(scheme)://\(hostname):\(port)/\(jobEndpoint)") else {
            throw NVFlareError.jobFetchFailed
        }
        
        // Prepare request body
        let requestBody: [String: Any] = [
            "capabilities": self.capabilities  // Use capabilities directly
        ]
        
        let body = try JSONSerialization.data(withJSONObject: requestBody)
        
        // Create request
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(deviceId, forHTTPHeaderField: "X-Flare-Device-ID")

        // Convert deviceInfo to JSON string
        let deviceInfoString = infoToQueryString(info: deviceInfo)
        request.setValue(deviceInfoString, forHTTPHeaderField: "X-Flare-Device-Info")

        // For now, sending empty user info
        request.setValue("", forHTTPHeaderField: "X-Flare-User-Info")
        request.httpBody = body
        
        print("Sending request: \(request.httpMethod!) \(request.url!)")
        print("Headers: \(request.allHTTPHeaderFields ?? [:])")
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            print("Error: Response is not HTTPURLResponse")
            throw NVFlareError.jobFetchFailed
        }
        print("Response Status Code: \(httpResponse.statusCode)")
        print("Response Headers: \(httpResponse.allHeaderFields)")
        // Print raw data as JSON for easy structure inspection
        if let jsonObject = try? JSONSerialization.jsonObject(with: data),
           let prettyData = try? JSONSerialization.data(withJSONObject: jsonObject, options: .prettyPrinted),
           let prettyString = String(data: prettyData, encoding: .utf8) {
            print("Response JSON Structure:")
            print(prettyString)
        } else {
            print("Raw Data (hex): \(data.map { String(format: "%02x", $0) }.joined())")
        }
        // First decode the response to get potential error messages
        let jobResponse = try JSONDecoder().decode(JobResponse.self, from: data)
        
        switch httpResponse.statusCode {
        case 200:
            switch jobResponse.status {
            case "OK":
                return jobResponse
            case "RETRY":
                if let retryWait = jobResponse.retryWait {
                    try await Task.sleep(for: .seconds(retryWait))
                    return try await fetchJob()  // Retry after waiting
                }
                throw NVFlareError.jobFetchFailed
            default:
                throw NVFlareError.jobFetchFailed
            }
            
        case 400:
            throw NVFlareError.invalidRequest(jobResponse.message ?? "Invalid request")
            
        case 403:
            throw NVFlareError.authError(jobResponse.message ?? "Authentication error")
            
        case 500:
            throw NVFlareError.serverError(jobResponse.message ?? "Server error")
            
        default:
            throw NVFlareError.jobFetchFailed
        }
    }
    
    func fetchTask(jobId: String) async throws -> TaskResponse {
        var urlComponents = URLComponents()
        urlComponents.scheme = scheme
        urlComponents.host = hostname
        urlComponents.port = port
        urlComponents.path = "/\(taskEndpoint)"
        urlComponents.queryItems = [
            URLQueryItem(name: "job_id", value: jobId)
        ]
        
        guard let url = urlComponents.url else {
            throw NVFlareError.taskFetchFailed("Could not construct URL")
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(deviceId, forHTTPHeaderField: "X-Flare-Device-ID")
        
        // Convert deviceInfo to JSON string
        if let deviceInfoData = try? JSONSerialization.data(withJSONObject: deviceInfo),
           let deviceInfoString = String(data: deviceInfoData, encoding: .utf8) {
            request.setValue(deviceInfoString, forHTTPHeaderField: "X-Flare-Device-Info")
        }
        
        // Empty user info for now
        request.setValue("{}", forHTTPHeaderField: "X-Flare-User-Info")
        
        let requestBody: [String: Any]

        if let cookie = currentCookie {
            requestBody = ["cookie": cookie.jsonObject]
        } else {
            requestBody = [:]  // empty dictionary means empty JSON object
        }

        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        
        print("Sending request: \(request.httpMethod!) \(request.url!)")
        print("Headers: \(request.allHTTPHeaderFields ?? [:])")
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw NVFlareError.taskFetchFailed("wrong response type")
        }
        
        print(httpResponse)
        
        // Decode response first to get potential error messages
        let taskResponse = try JSONDecoder().decode(TaskResponse.self, from: data)
        
        switch httpResponse.statusCode {
        case 200:
            switch taskResponse.status {
            case "OK":
                // TODO:: check taskResponse cookie somehow regardless of status
                currentCookie = taskResponse.cookie
                return taskResponse
            case "DONE":
                return taskResponse
            case "RETRY":
                if let retryWait = taskResponse.retryWait {
                    try await Task.sleep(for: .seconds(retryWait))
                    return try await fetchTask(jobId: jobId)
                }
                throw NVFlareError.taskFetchFailed("Retry RETRY Failed")
            case "NO_TASK":
                if let retryWait = taskResponse.retryWait {
                    try await Task.sleep(for: .seconds(retryWait))
                    return try await fetchTask(jobId: jobId)
                }
                throw NVFlareError.taskFetchFailed("Retry NO_TASK Failed")
            default:
                throw NVFlareError.taskFetchFailed("Wrong Status: \(taskResponse.status)")
            }
            
        case 400:
            throw NVFlareError.invalidRequest(taskResponse.message ?? "Invalid request")
            
        case 403:
            throw NVFlareError.authError(taskResponse.message ?? "Authentication error")
            
        case 500:
            throw NVFlareError.serverError(taskResponse.message ?? "Server error")
            
        default:
            throw NVFlareError.taskFetchFailed("Wrong Status Code: \(httpResponse.statusCode)")
        }
    }
    
    func sendResult(jobId: String, taskId: String, taskName: String, weightDiff: [String: Any]) async throws {
        var urlComponents = URLComponents()
        urlComponents.scheme = scheme
        urlComponents.host = hostname
        urlComponents.port = port
        urlComponents.path = "/\(resultEndpoint)"
        urlComponents.queryItems = [
            URLQueryItem(name: "job_id", value: jobId),
            URLQueryItem(name: "task_id", value: taskId),
            URLQueryItem(name: "task_name", value: taskName)
        ]
        
        guard let url = urlComponents.url else {
            throw NVFlareError.trainingFailed("Invalid URL")
        }
        
        // Create request
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(deviceId, forHTTPHeaderField: "X-Flare-Device-ID")
        
        // Convert deviceInfo to JSON string
        if let deviceInfoData = try? JSONSerialization.data(withJSONObject: deviceInfo),
           let deviceInfoString = String(data: deviceInfoData, encoding: .utf8) {
            request.setValue(deviceInfoString, forHTTPHeaderField: "X-Flare-Device-Info")
        }
        
        // Empty user info for now
        request.setValue("{}", forHTTPHeaderField: "X-Flare-User-Info")
        
        guard let cookie = currentCookie else {
            throw NVFlareError.authError("No cookie found")
        }
        
        // Prepare request body
        let requestBody: [String: Any] = [
            "result": weightDiff,
            "cookie": cookie.jsonObject
        ]
        
        // Serialize to JSON
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        
        print("Sending request: \(request.httpMethod!) \(request.url!)")
        print("Headers: \(request.allHTTPHeaderFields ?? [:])")
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw NVFlareError.trainingFailed("Invalid response")
        }
        
        switch httpResponse.statusCode {
        case 200:
            let resultResponse = try JSONDecoder().decode(ResultResponse.self, from: data)
            
            switch resultResponse.status {
            case "OK":
                return  // Success
                
            case "INVALID":
                throw NVFlareError.trainingFailed(resultResponse.message ?? "Invalid result")
                
            default:
                throw NVFlareError.trainingFailed("Unknown status")
            }
            
        case 400:
            let errorResponse = try JSONDecoder().decode(ResultResponse.self, from: data)
            throw NVFlareError.invalidRequest(errorResponse.message ?? "Invalid request")
            
        case 403:
            let errorResponse = try JSONDecoder().decode(ResultResponse.self, from: data)
            throw NVFlareError.authError(errorResponse.message ?? "Authentication error")
            
        case 500:
            let errorResponse = try JSONDecoder().decode(ResultResponse.self, from: data)
            throw NVFlareError.serverError(errorResponse.message ?? "Server error")
            
        default:
            throw NVFlareError.trainingFailed("Unexpected status code: \(httpResponse.statusCode)")
        }
    }
}
