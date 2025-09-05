## NVFlareSDK Usage
First, copy the whole folder into your app, please refer to ExampleProject on how to do that.
                                                                                                                                                                                                             
## API Usage

```
let dataSource = SimpleDataSource(dataset: NVFlareDataset)
let runner = try NVFlareRunner(
    jobName: selectedJob.rawValue,
    dataSource: dataSource,
    deviceInfo: [
          "device_id": UIDevice.current.identifierForVendor?.uuidString ?? "unknown",
         "platform": "ios",
        "app_version": Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "unknown"
    ],
    userInfo: [:],
    jobTimeout: 30.0,
    serverURL: "https://\(serverHost):\(serverPort)",
    allowSelfSignedCerts: true  // Set to true for self-signed HTTPS certificates
)
```
