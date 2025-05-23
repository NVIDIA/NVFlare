@objc protocol Trainer {
    func train() async throws -> [String: Any]
}
