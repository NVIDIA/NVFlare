//
//  ContentView.swift
//  NVFlareMobile
//
//

import SwiftUI
import SystemConfiguration
import UIKit

struct ContentView: View {
    @StateObject private var trainerController: TrainerController
    @StateObject private var connection: Connection
    
    init() {
        //let connection = Connection(hostname: "192.168.6.101", port: 4321)
        let connection = Connection(hostname: "169.254.39.153", port: 4321)
        _connection = StateObject(wrappedValue: connection)
        _trainerController = StateObject(wrappedValue: TrainerController(connection: connection))
    }
    
    var body: some View {
        VStack {
            TextField("Hostname", text: $connection.hostname)
                .textFieldStyle(RoundedBorderTextFieldStyle())
            
            TextField("Port", value: $connection.port, formatter: NumberFormatter())
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .keyboardType(.numberPad)
            
            Picker("Trainer Type", selection: $trainerController.trainerType) {
                ForEach(TrainerType.allCases, id: \.self) { type in
                    Text(type.rawValue).tag(type)
                }
            }
            .pickerStyle(SegmentedPickerStyle())
            
            VStack(alignment: .leading, spacing: 10) {
                Text("Supported Methods")
                    .font(.headline)
                    .padding(.bottom, 4)
                
                ScrollView {
                    VStack(alignment: .leading, spacing: 8) {
                        ForEach(MethodType.allCases, id: \.self) { method in
                            Toggle(method.displayName, isOn: Binding(
                                get: { trainerController.supportedMethods.contains(method) },
                                set: { _ in trainerController.toggleMethod(method) }
                            ))
                        }
                    }
                    .padding(.horizontal)
                }
                .frame(maxHeight: 200)  // Limit height and make scrollable
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(10)
            
            Button(trainerController.status == .training ? "Stop Training" : "Start Training") {
                if trainerController.status == .training {
                    trainerController.stopTraining()
                } else {
                    Task {
                        do {
                            try await trainerController.startTraining()
                            if trainerController.status == .training {
                                trainerController.status = .idle
                            }
                        } catch {
                            print("Training error: \(error)")
                            trainerController.status = .idle
                        }
                    }
                }
            }
            .disabled(trainerController.status == .stopping || trainerController.supportedMethods.isEmpty)
            
            if trainerController.status == .training {
                ProgressView()
            }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
