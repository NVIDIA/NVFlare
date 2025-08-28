//
//  NVFlareError.swift
//  NVFlare iOS SDK
//
//  Error definitions for the NVFlare SDK
//

import Foundation

/// NVFlare SDK error types
public enum NVFlareError: Error {
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

/// Dataset-related error types
public enum DatasetError: Error {
    case noDataFound
    case dataLoadFailed
    case invalidDataFormat
    case emptyDataset
} 