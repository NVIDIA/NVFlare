This is a simple example to show how to build NVFlare docker image.


# NVFlare application code package
We can pre-install custom codes inside each docker image.
We utilize our nvflare pre-install command to do that.

First, we need to prepare the `application_code.zip` folder structure:

```bash
application_code_folder
├── application/                    # optional
│   └── <job_name>/
│               ├── meta.json       # job metadata
│               ├── app_<site>/     # Site custom code
│                  └── custom/      # Site custom code
├── application-share/              # Shared resources
|   └── simple_network.py           # Shared model definition 
└── requirements.txt       # Python dependencies (optional)
```

We have already prepared application-share folder and requirements.txt in this example.
We run the following command to create a zip folder so we can use that to build the CVM:

```bash
python -m zipfile -c application_code.zip application_code/*
```

# NVFlare docker file

We have prepared a `Dockerfile`.
Please run the following command to build the NVFlare docker image:

```bash
./docker_build.sh Dockerfile nvflare-site 
```
