# Research Directory
Research directory will be the place to host various research work from community on Federated learning
leveraging NVIDIA FLARE. **The code will not be maintained by NVIDIA FLARE team**, but will require Pull Request
approval process. 

## License
By providing the code in NVFLARE repository, you will grant the research project in NVIDIA repo to be released under open source license
Apache v2 License or equivalent open source license

## Requirements
Each research project should create a sub directory with following requirements

* sub directory name must be in ASCII string, all in lower, kebab-case and no longer than 35 characters long
* Each project should include
  * README.md -- document must include
    * objective 
    * background
    * description
    * nvflare version used
    * data download and preparation ( if applicable )
    * steps to run the code
    * expected results
  * all code should be in runnable condition, i.e. no broken code
  * License file

## Example
```
sample_research$ 
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
```
