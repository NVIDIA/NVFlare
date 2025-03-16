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

    // TODO:: parse from the right server response
    init(from data: [String: Any], method: String) {
        self.totalEpochs = data["total_epochs"] as? Int ?? 1
        self.batchSize = (method == "xor") ? 1 : 32  // Set batch size based on method
        self.learningRate = data["learning_rate"] as? Float ?? 0.1
        self.method = method
        self.dataSetType = data["dataset_type"] as? String ?? method  // Default to method if not specified
    }
}

struct Job {
    let id: String
    let meta: JobMeta
    let status: String
}
