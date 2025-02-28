import Foundation
import UIKit
import SystemConfiguration

@MainActor
class DeviceStateMonitor: ObservableObject {
    @Published private(set) var isReadyForTraining = false
    
    private var thermalStateObserver: NSObjectProtocol?
    private var batteryStateObserver: NSObjectProtocol?
    
    init() {
        setupInitialState()
    }
    
    private func setupInitialState() {
        UIDevice.current.isBatteryMonitoringEnabled = true
        setupObservers()
        checkDeviceState()
    }
    
    deinit {
        if let thermalObserver = thermalStateObserver {
            NotificationCenter.default.removeObserver(thermalObserver)
        }
        if let batteryObserver = batteryStateObserver {
            NotificationCenter.default.removeObserver(batteryObserver)
        }
    }
    
    private func setupObservers() {
        thermalStateObserver = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.checkDeviceState()
            }
        }
        
        batteryStateObserver = NotificationCenter.default.addObserver(
            forName: UIDevice.batteryStateDidChangeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.checkDeviceState()
            }
        }
    }
    
    private var isWiFiConnected: Bool {
        guard let reachability = SCNetworkReachabilityCreateWithName(kCFAllocatorDefault, "www.apple.com") else {
            return false
        }
        var flags = SCNetworkReachabilityFlags()
        SCNetworkReachabilityGetFlags(reachability, &flags)
        return flags.contains(.reachable) && !flags.contains(.isWWAN)
    }
    
    private func checkDeviceState() {
        let isCharging = UIDevice.current.batteryState == .charging ||
                        UIDevice.current.batteryState == .full
        
        let isNotHot = ProcessInfo.processInfo.thermalState != .critical &&
                       ProcessInfo.processInfo.thermalState != .serious
        
        let isIdle = !ProcessInfo.processInfo.isLowPowerModeEnabled
        
        let batteryLevel = UIDevice.current.batteryLevel >= 0.5 // 50% or more
        
        isReadyForTraining = isCharging && isNotHot && isIdle && 
                            batteryLevel && isWiFiConnected
    }
}
