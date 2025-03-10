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
    let dataSetType: String
    
    var dictionary: [String: Any] {
        return [
            "total_epochs": totalEpochs,
            "batch_size": batchSize,
            "learning_rate": learningRate,
            "dataset_type": dataSetType
        ]
    }
    
    init(from data: [String: Any]) {
        self.totalEpochs = data["total_epochs"] as? Int ?? 1
        self.batchSize = data["batch_size"] as? Int ?? 1
        self.learningRate = data["learning_rate"] as? Float ?? 0.1
        self.dataSetType = data["dataset_type"] as? String ?? "xor"
    }
}

struct Job {
    let id: String
    let meta: JobMeta
    let status: String
}
