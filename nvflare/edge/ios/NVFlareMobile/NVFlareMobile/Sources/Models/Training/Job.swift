//
//  Job.swift
//  NVFlareMobile
//
//

import Foundation

struct JobMeta {
    let totalEpochs: Int
    let batchSize: Int
    let learningRate: Float
    let method: String
    var dataSetType: String  // Derived from method, kept for compatibility
    
    var dictionary: [String: Any] {
        return [
            "total_epochs": totalEpochs,
            "batch_size": batchSize,
            "learning_rate": learningRate,
            "method": method,
            "dataset_type": dataSetType
        ]
    }
    
    init(from data: [String: Any]) {
        self.totalEpochs = data["total_epochs"] as? Int ?? 1
        self.batchSize = data["batch_size"] as? Int ?? 1
        self.learningRate = data["learning_rate"] as? Float ?? 0.1
        self.method = data["method"] as? String ?? "xor"
        self.dataSetType = data["dataset_type"] as? String ?? self.method  // Default to method if not specified
    }
}

struct Job {
    let id: String
    let meta: JobMeta
    let status: String
}
