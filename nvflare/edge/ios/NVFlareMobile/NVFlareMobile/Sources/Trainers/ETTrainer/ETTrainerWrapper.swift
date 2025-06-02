//
//  ETTrainerWrapper.swift
//  NVFlareMobile
//
//


class ETTrainerWrapper: Trainer {
    private let etTrainer: ETTrainer
    
    init(modelBase64: String, meta: TrainingConfig) {
        print("ETTrainerWrapper: Initializing with model and meta")
        self.etTrainer = ETTrainer(modelBase64: modelBase64, meta: meta.dictionary)
        print("ETTrainerWrapper: Initialization complete")
    }
    
    func train() async throws -> [String: Any] {
        print("ETTrainerWrapper: Starting train()")
        let result = etTrainer.train()
        print("ETTrainerWrapper: train() completed with result keys: \(result.keys)")
        return result
    }
}
