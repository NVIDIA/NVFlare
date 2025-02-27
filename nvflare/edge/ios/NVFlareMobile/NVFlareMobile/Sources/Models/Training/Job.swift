//
//  Job.swift
//  NVFlareMobile
//
//  Created by Yuan-Ting Hsieh on 2/26/25.
//

import Foundation


struct JobMeta {
    let totalEpochs: Int
    let batchSize: Int
    let learningRate: Float
    
    var dictionary: [String: Any] {
        return [
            "total_epochs": totalEpochs,
            "batch_size": batchSize,
            "learning_rate": learningRate
        ]
    }
    
    init?(from data: [String: Any]) {
        guard let totalEpochs = data["total_epochs"] as? Int,
              let batchSize = data["batch_size"] as? Int,
              let learningRate = data["learning_rate"] as? Float else {
            return nil
        }
        self.totalEpochs = totalEpochs
        self.batchSize = batchSize
        self.learningRate = learningRate
    }
    
    init(from response: JobDataResponse) {
        self.totalEpochs = response.totalEpochs
        self.batchSize = response.batchSize
        self.learningRate = response.learningRate
    }
}

struct Job {
    let id: String
    let meta: JobMeta
    let status: String
}
