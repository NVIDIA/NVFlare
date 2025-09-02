//
//  NVFlareConnection.swift
//  NVFlare iOS SDK
//
//  Server connection and communication for NVFlare federated learning
//

import Foundation
import UIKit

/// URLSession delegate for handling self-signed certificates
private class SelfSignedCertDelegate: NSObject, URLSessionDelegate {
    func urlSession(_ session: URLSession, didReceive challenge: URLAuthenticationChallenge, completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void) {
        
        // Check if this is a server trust challenge
        guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust else {
            completionHandler(.performDefaultHandling, nil)
            return
        }
        
        // For self-signed certificates, create credential with server trust
        if let serverTrust = challenge.protectionSpace.serverTrust {
            let credential = URLCredential(trust: serverTrust)
            print("NVFlareConnection: Accepting self-signed certificate for \(challenge.protectionSpace.host)")
            completionHandler(.useCredential, credential)
        } else {
            completionHandler(.performDefaultHandling, nil)
        }
    }
}

/// Main connection class for communicating with NVFlare server
public class NVFlareConnection: ObservableObject {
    @Published public var serverURL: String = ""
    
    private let deviceId = UIDevice.current.identifierForVendor?.uuidString ?? "unknown"
    private let configuredDeviceInfo: [String: String]
    private let urlSession: URLSession
    
    // Device info passed from NVFlareRunner or default minimal info
    private var deviceInfo: [String: String] {
        return configuredDeviceInfo.isEmpty ? [
            "device_id": deviceId,
            "platform": "ios",
            "app_version": Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "unknown"
        ] : configuredDeviceInfo
    }

    private let jobEndpoint = "job"
    private let taskEndpoint = "task"
    private let resultEndpoint = "result"
    private var capabilities: [String: Any] = ["methods": []]
    private var currentCookie: JSONValue?

    public var isValid: Bool {
        return !serverURL.isEmpty && URL(string: serverURL) != nil
    }
    
    public init(serverURL: String, deviceInfo: [String: String] = [:], allowSelfSignedCerts: Bool = false) {
        self.serverURL = serverURL
        self.configuredDeviceInfo = deviceInfo
        
        // Create URLSession configuration
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30.0
        config.timeoutIntervalForResource = 60.0
        
        // Create URLSession with or without delegate for self-signed certificates
        if allowSelfSignedCerts {
            self.urlSession = URLSession(configuration: config, delegate: SelfSignedCertDelegate(), delegateQueue: nil)
        } else {
            self.urlSession = URLSession(configuration: config)
        }
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
    
    func fetchJob(jobName: String) async throws -> JobResponse {
        guard let url = URL(string: "\(serverURL)/\(jobEndpoint)") else {
            throw NVFlareError.jobFetchFailed
        }
        
        // Prepare request body
        let requestBody: [String: Any] = [
            "capabilities": ["methods": [jobName]],
            "job_name": jobName,
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
        
        let (data, response) = try await urlSession.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            print("Error: Response is not HTTPURLResponse")
            throw NVFlareError.jobFetchFailed
        }
        print("Response Status Code: \(httpResponse.statusCode)")
        print("Response Headers: \(httpResponse.allHeaderFields)")
        
        if let responseString = String(data: data, encoding: .utf8) {
            print("Raw Response Body: \(responseString)")
        }
        
        guard httpResponse.statusCode == 200 else {
            print("HTTP Error: Status code \(httpResponse.statusCode)")
            throw NVFlareError.jobFetchFailed
        }
        
        do {
            let jobResponse = try JSONDecoder().decode(JobResponse.self, from: data)
            print("Decoded JobResponse: \(jobResponse)")
            return jobResponse
        } catch {
            print("JSON Decode Error: \(error)")
            if let responseString = String(data: data, encoding: .utf8) {
                print("Failed to decode response: \(responseString)")
            }
            throw NVFlareError.jobFetchFailed
        }
    }
    
    func fetchTask(jobId: String) async throws -> TaskResponse {
        guard let baseURL = URL(string: serverURL) else {
            throw NVFlareError.taskFetchFailed("Invalid server URL")
        }
        
        var urlComponents = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)
        urlComponents?.path = "/\(taskEndpoint)"
        urlComponents?.queryItems = [
            URLQueryItem(name: "job_id", value: jobId)
        ]
        
        guard let url = urlComponents?.url else {
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
            requestBody = [:]
        }

        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        
        print("Sending request: \(request.httpMethod!) \(request.url!)")
        print("Headers: \(request.allHTTPHeaderFields ?? [:])")
        
        let (data, response) = try await urlSession.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw NVFlareError.taskFetchFailed("wrong response type")
        }
        
        print(httpResponse)
        
        if let responseString = String(data: data, encoding: .utf8) {
            print("Raw Response Body: \(responseString)")
        }
        
        guard httpResponse.statusCode == 200 else {
            print("HTTP Error: Status code \(httpResponse.statusCode)")
            throw NVFlareError.taskFetchFailed("HTTP \(httpResponse.statusCode)")
        }
        
        do {
            let taskResponse = try JSONDecoder().decode(TaskResponse.self, from: data)
            print("Decoded TaskResponse: \(taskResponse)")
            
            // Store cookie for later use in result reporting
            currentCookie = taskResponse.cookie
            
            return taskResponse
        } catch {
            print("JSON Decode Error: \(error)")
            if let responseString = String(data: data, encoding: .utf8) {
                print("Failed to decode response: \(responseString)")
            }
            throw NVFlareError.taskFetchFailed("decode error: \(error)")
        }
    }
    
    func sendResult(jobId: String, taskId: String, taskName: String, weightDiff: [String: Any]) async throws -> ResultResponse {
        guard let baseURL = URL(string: serverURL) else {
            throw NVFlareError.trainingFailed("Invalid server URL")
        }
        
        var urlComponents = URLComponents(url: baseURL, resolvingAgainstBaseURL: false)
        urlComponents?.path = "/\(resultEndpoint)"
        urlComponents?.queryItems = [
            URLQueryItem(name: "job_id", value: jobId),
            URLQueryItem(name: "task_id", value: taskId),
            URLQueryItem(name: "task_name", value: taskName)
        ]
        
        guard let url = urlComponents?.url else {
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
        
        let (data, response) = try await urlSession.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw NVFlareError.trainingFailed("Invalid response")
        }
        
        print("Response Status Code: \(httpResponse.statusCode)")
        if let responseString = String(data: data, encoding: .utf8) {
            print("Response Body: \(responseString)")
        }
        
        // Parse the response regardless of status code - FSM needs to handle different statuses
        do {
            let resultResponse = try JSONDecoder().decode(ResultResponse.self, from: data)
            print("Decoded ResultResponse: \(resultResponse)")
            return resultResponse
        } catch {
            print("Failed to decode ResultResponse: \(error)")
            throw NVFlareError.trainingFailed("decode error: \(error)")
        }
    }
}
