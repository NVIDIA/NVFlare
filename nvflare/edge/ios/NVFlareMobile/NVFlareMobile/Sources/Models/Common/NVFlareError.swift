//
//  NVFlareError.swift
//  NVFlareMobile
//
//  Created by Yuan-Ting Hsieh on 2/26/25.
//

import Foundation

enum NVFlareError: Error {
    // Network related
    case jobFetchFailed
    case taskFetchFailed
    case invalidRequest(String)
    case authError(String)
    case serverError(String)
    
    // Training related
    case invalidMetadata
    case invalidModelData
    case trainingFailed(String)
    case serverRequestedStop
}
