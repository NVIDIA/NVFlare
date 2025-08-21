//
//  NVFlareConstants.swift
//  NVFlareSDK
//
//  Status constants for NVFlare iOS SDK
//

import Foundation

/// Job status constants - used for job lifecycle management
public enum JobStatus {
    public static let stopped = "stopped"
    public static let done = "DONE"
    public static let running = "RUNNING"
    public static let submitted = "SUBMITTED"
    public static let approved = "APPROVED"
    public static let dispatched = "DISPATCHED"
}

/// Note: Task status constants are defined in TaskResponse.TaskStatus enum
/// Additional edge API status constants that don't fit in the task enum
public enum EdgeApiStatus {
    public static let noJob = "NO_JOB"
    public static let noTask = "NO_TASK"
    public static let invalidRequest = "INVALID_REQUEST"
}
