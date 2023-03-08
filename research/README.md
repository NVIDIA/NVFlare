# Research Directory
This research directory is the place to host various research work from the community on Federated learning
leveraging NVIDIA FLARE. **The code will not be maintained by NVIDIA FLARE team**, but will require Pull Request
approval process. 

## License
By providing the code in NVFLARE repository, you will grant the research project in NVIDIA repo to be released under Apache v2 License or equivalent open source license.

## Requirements
Each research project should create a subdirectory with the following requirements.

* Subdirectory name must be in ASCII string, all in lower, kebab-case, and no longer than 35 characters long
* Each project should include
  * README.md -- document must include
    * Objective 
    * Background
    * Description
    * Setup
    * Steps to run the code 
    * Data download and preparation (if applicable)
    * Expected results
  * Jobs-folder including configurations and optional custom code
  * All code should be in runnable condition, i.e., no broken code
  * License file
  * Requirements file listing all dependencies, including the NVFLARE version used

## Example
```
sample_research$ 
.
├── jobs
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
└── LICENSE
└── requirements.txt
```

## Setup
To run the research code, we recommend using a virtual environment.

### Set up a virtual environment
```
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv
```
(If needed) make all shell scripts executable using
```
find . -name ".sh" -exec chmod +x {} \;
```
initialize virtual environment.
```
python3 -m venv venv
source venv/bin/activate
```
within each research folder, install required packages for training
```
pip install --upgrade pip
pip install -r requirements.txt
```
