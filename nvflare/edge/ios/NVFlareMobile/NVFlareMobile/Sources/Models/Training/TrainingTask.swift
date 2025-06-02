//
//  TrainingTask.swift
//  NVFlareMobile
//
//

import Foundation

struct TrainingConfig {
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

    // Default initializer
    init() {
        self.totalEpochs = 1
        self.batchSize = 4
        self.learningRate = 0.1
        self.method = "cnn"
        self.dataSetType = "cifar10"
    }
    
    init(from data: [String: Any]) {
        self.totalEpochs = data["total_epochs"] as? Int ?? 1
        self.batchSize = data["batch_size"] as? Int ?? 4
        self.learningRate = data["learning_rate"] as? Float ?? 0.1
        self.method = data["method"] as? String ?? "xor"
        self.dataSetType = data["dataset_type"] as? String ?? "xor"
    }
}

struct TrainingTask {
    let id: String
    let name: String
    let jobId: String
    let modelData: String
    let trainingConfig: TrainingConfig
}
