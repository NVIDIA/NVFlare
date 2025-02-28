//
//  ETTrainerWrapper.swift
//  NVFlareMobile
//
//  Created by Yuan-Ting Hsieh on 2/27/25.
//


class ETTrainerWrapper: Trainer {
    private let etTrainer: ETTrainer
    
    init(modelBase64: String, meta: JobMeta) {
        self.etTrainer = ETTrainer(modelBase64: modelBase64, meta: meta.dictionary)
    }
    
    func train() async throws -> [String: Any] {
        return etTrainer.train()
    }
}
