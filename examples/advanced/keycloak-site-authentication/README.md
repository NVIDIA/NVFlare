# Example for Job-level authorization

# Overview

The purpose of this example is to demonstrate following features of NVFlare,

1. Run NVFlare in secure mode
2. Demonstrate the ability to integrate with KeyCloak third-party user authentication for managing the NVFlare system. 

## System Requirements

1. Install Python and set up a Virtual Environment,
```
python3 -m venv nvflare-env
source nvflare-env/bin/activate
```
2. Install NVFlare
```
pip install nvflare
```
3. The example is part of the NVFlare source code. The source code can be obtained like this,
```
git clone https://github.com/NVIDIA/NVFlare.git
```
4. TLS requires domain names. Please add the following line in the `/etc/hosts` file,
```
127.0.0.1	server1
```

### Setup

```
nvflare poc prepare -i project.yml -c site_a site_b
```

All the startup kits will be generated in this folder,
```
/tmp/nvflare/poc/keycloak-site-authentication/prod_00
```

### Set up KeyCloak

Download and install the KeyCloak following the instruction guide from https://www.keycloak.org/getting-started/getting-started-zip. Key steps for the installation:

* Start the KeyCLoak by running "bin/kc.sh start-dev"
* Set up the realm called "myrealm"
* Setup the user "myuser@example.com"
* Add the client "myclient"
* The KeyCLoak will be running at http://localhost:8080/

### Set up Client side authentication requirement

```
cp -r site/local/* /tmp/nvflare/poc/keycloak-site-authentication/prod_00/site_a/local/* 
```
This will set up site_a with additional user security authentication to the KeyCloak.

### Set up Admin user authentication

```
cp -r admin/local/* /tmp/nvflare/poc/keycloak-site-authentication/prod_00/myuser@example.co/local/* 
```
This will set up admin user myuser@example.com to acquire KeyCloak access_token when starting the admin tool.


### Starting NVFlare

This script will start up the server and 2 clients,
```
nvflare poc start
```

### Logging with Admin Console

Start the myuser@example.com admin tool and login as the `myuser@example.com` user. Also provide the password to the KeyCloak:

```
cd /tmp/nvflare/poc/job-level-authorization/prod_00/myuser@example.com
./startup/fl_admin.sh
```

At the prompt, enter the user email `myuser@example.com`, and then provide the password to `site_a` KeyCloak.


### User authentication when running jobs

With this system set up, the `aite_a` will require additional user authentication when submitting and running a job. `site_b` does not have this additional security requirement. Any admin user can submit and run the job.

Let's choose the `hello-numpy-sag` job from the `hello-world` examples. For demonstrating purpose, let's change the `min_clients` in the job meta.json to 1.  

#### Authenticated admin user

* `myuser@example.com` corrected authenticated to `site_a` KeyCloak system. When he submitting the job, the `hello-numpy-sag` job is successfully running on both `site_a` and `site_b`.

#### Un-authenticated admin user

* If the `myuser@example.com` admin user provides the wrong password, or for some reason KeyCloak system is not available when starting the admin tool, or submitting the job, the `hello-numpy-sag` job won't be able to run the `site_a`. 
* `site_a` will show "ERROR - Authorization failed".
* The job can successfully run on `site-b`.
* `list_jobs -d JOB_ID` command will show "job_deploy_detail" information of this job.


