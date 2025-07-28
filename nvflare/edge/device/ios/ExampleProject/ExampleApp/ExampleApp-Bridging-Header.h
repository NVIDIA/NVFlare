//
//  Use this file to import your target's public headers that you would like to expose to Swift.
//

#import "../NVFlareSDK/Training/ETTrainer/ETTrainer.h"
#import "../NVFlareSDK/Models/Common/Constants.h"

// App's C++ dataset creation functions
void* CreateAppCIFAR10Dataset(void);
void* CreateAppXORDataset(void);
void DestroyAppDataset(void* dataset);
