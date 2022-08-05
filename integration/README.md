# Integration Directory
Integration directory will be the place to host various NVIDIA FLARE integrations with other frameworks.
**The code will may not be maintained by NVIDIA FLARE team**, but will require Pull Request
approval process. 

## Requirements
Each research project should create a sub directory with following requirements

* sub directory name must be in ASCII string and no longer than 35 characters long
* Each project should include
  * README.md -- document must include
    * objective 
    * background
    * description
    * nvflare version used
    * data download and preparation
    * steps to run the code
    * expected results
  * code needed to run nvflare job or Pull Request
  * license file

## Example
```
sample$ 
.
├── job_configs
    └── job1
           ├── app_server
                   ├── config
                           └── config_fed_server.json
                   └── custom
                        └── sample_controller.py
           └── app_client
                   ├── config
                           └── config_fed_client.json
                   └── custom
                        └── sample_executor.py
          └── meta.json
└── README.md
└── license.txt
