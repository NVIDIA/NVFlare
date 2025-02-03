# Example of site authentication integration with keycloak

# Overview

In this example, We demonstrate the NVFLARE's capability of supporting site-specific authentication, using KeyCloak integration as example.

NVFLARE is agnostic to the 3rd party authentication mechanism, each client can have its own authentication system, user can replace KeyCloak with LDAP or any other authentication systems.

## Getting Started: Quick Start

### Install NVFLARE

```
python3 -m pip install nvflare
```

Clone NVFLARE repo to get examples, switch main branch (latest stable branch)

```
git clone https://github.com/NVIDIA/NVFlare.git
cd NVFlare
git switch main
```

### Setup

```
cd examples/advanced/keycloak-site-authentication
nvflare poc prepare -i project.yml -c site_a site_b
```

All the startup kits will be generated in this folder,
```
/tmp/nvflare/poc/keycloak-site-authentication/prod_00
```

### Set up KeyCloak

Download and install the KeyCloak following the instruction guide from https://www.keycloak.org/getting-started/getting-started-zip. Key steps for the installation:

<!-- markdown-link-check-disable -->

* Start the KeyCLoak by running "bin/kc.sh start-dev"
* Set up the realm called "myrealm"
* Setup the user "myuser@example.com"
* Add the client "myclient"
* The KeyCLoak will be running at http://localhost:8080/

### Set up FL Client Job Authorization Requirement

```
cp -r site/local/* /tmp/nvflare/poc/keycloak-site-authentication/prod_00/site_a/local/* 
```

Save the KeyCloak public_key in the `/tmp/nvflare/poc/keycloak-site-authentication/prod_00/site_a/local/public_key.pem` file, with the following format:

```
-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAre3kQxqOfTJ7LLRwlpotw47goqSsuyFOg9Ihx5IXDMbO8HTGuGQcdDVJaYJQYphfhp2qdw+1o6qVN2yPBxwiBWju/XZQMPbCXRBu2bVDffWJVMoelLDbr3uY9hCgYgmB7qYpDdNOmxb2+xIlg/x0q+vrRRMtdd8SGicvjg0mQSEEF4a7QOSwuDnwBX8+bMOXfyB5qQJlakNVND1Bc+MjDENkHLtImVowX9XZcz8M6Ap9Eq1z2agl6lmFxTLtZroTE6IQS/dFYPVy4rZ1Zuy5cvs/3j+SYzlplH/iP3qZs8UiKrTJMmfIuLmDbP3hEAOsEmQ/M3lRxnE4wuGxvel5rwIDAQAB
-----END PUBLIC KEY-----
```

In the local/custom/resources.json config file, it contains the following additional security handler:

```
    {
      "id": "security_handler",
      "path": "keycloak_security_handler.CustomSecurityHandler"
    }
```

The CustomSecurityHandler in the custom/keycloak_security_handler.py contains the logic to validate the admin user's KeyCloak access token when the admin user submits a job, or scheduler picks up an already submitted job from the admin user. If the access token is invalid, the job will not be authorized to run.

### Set up Admin user authentication

```
cp -r admin/local/* /tmp/nvflare/poc/keycloak-site-authentication/prod_00/myuser@example.co/local/* 
```

In the local/custom/resources.json config file, it contains the following admin event handler. the "orgs" arg provides a list of site names, and it's corresponding KeyCloak access_token URLs:

```
      {
        "id": "auth",
        "path": "admin_auth.AdminAuth",
        "args": {
          "orgs": {
            "site-a": "http://localhost:8080/realms/myrealm/protocol/openid-connect/token"
          }
        }
      }
```

The AdminAuth event handler in the custom/admin_auth.py has the logic to acquire the KeyCloak access tokens to each individual site. When the admin user submits a job, it will set the tokens in the FLContext.

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


### Require authenticated admin user when running jobs

With this system set up, the `site_a` will require only the authenticated admin user to be able to submit and run a job. `site_b` does not have this additional security requirement. Any admin user can submit and run the job.

Let's choose the `hello-numpy-sag` job from the `hello-world` examples. For demonstrating purpose, let's change the `min_clients` in the job meta.json to 1.  

#### Authenticated admin user

* `myuser@example.com` is successfully authenticated to `site_a` KeyCloak system. The `hello-numpy-sag` job is successfully submitted and run on both `site_a` and `site_b`.

#### Un-authenticated admin user

* If the `myuser@example.com` admin user provides the wrong password, or for some reason KeyCloak system is not available when starting the admin tool, or submitting the job, the `hello-numpy-sag` job won't be able to run the `site_a`. 
* `site_a` will show "ERROR - Authorization failed".
* The job can successfully run on `site-b`.
* `list_jobs -d JOB_ID` command will show "job_deploy_detail" information of this job.

<!-- markdown-link-check-enable -->
