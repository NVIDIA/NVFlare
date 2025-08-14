//
//  NVFlareError.swift
//  NVFlare iOS SDK
//
//  Error definitions for the NVFlare SDK
//

import Foundation

enum NVFlareError: Error {
    // Network related
    case jobFetchFailed
    case taskFetchFailed(String)
    case invalidRequest(String)
    case authError(String)
    case serverError(String)
    
    // Training related
    case invalidMetadata(String)
    case invalidModelData(String)
    case trainingFailed(String)
    case serverRequestedStop
} 