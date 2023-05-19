.. _dashboard_api:

#########################
Dashboard in NVIDIA FLARE
#########################
As mentioned in :ref:`provisioning`, the NVIDIA FLARE system requires a set of startup kits
which include the private keys and certificates (signed by the root CA) in order to communicate to one another.
The new :ref:`nvflare_dashboard_ui` in NVIDIA FLARE provides a simple way to collect information of clients and users from different organizations,
as well as to generate those startup kits for users to download.

Most of the details about provisioning can be found in :ref:`provisioning`.  In this section, we focus on the user interaction with Dashboard and its backend API.

*****************************
Dashboard commandline options
*****************************

Running ``nvflare dashboard -h`` shows all available options.

.. code-block:: shell

    (nvflare_venv) ~/workspace/repos/flare$ nvflare dashboard -h
    usage: nvflare dashboard [-h] [--start] [--stop] [-p PORT] [-f FOLDER] [-i DASHBOARD_IMAGE] [--passphrase PASSPHRASE] [-e ENV]

    optional arguments:
    -h, --help            show this help message and exit
    --start               start dashboard
    --stop                stop dashboard
    -p PORT, --port PORT  port to listen
    -f FOLDER, --folder FOLDER
                            folder containing necessary info (default: current working directory)
    --passphrase PASSPHRASE
                            Passphrase to encrypt/decrypt root CA private key. !!! Do not share it with others. !!!
    -e ENV, --env ENV     additonal environment variables: var1=value1

To start Dashboard, run ``nvflare dashboard --start``.

The Dashboard Docker will detect if the database is initialized.  If not, it will ask for the project_admin email address and will generate a random password:

.. code-block::

    Please provide project admin email address.  This person will be the super user of the dashboard and this project.
    project_admin@admin_organization.com
    generating random password
    Project admin credential is project_admin@admin_organization.com and the password is EXAMPLE1

Please log in with this credential to finish setting up the project in Dashboard once the system is up and running.
The project_admin can change his/her password in the Dashboard system after logging in.

Note that for the first time, it may take a while to download the nvflare image as you see the prompt:

.. code-block::

    Pulling nvflare/nvflare, may take some time to finish.

After pulling the image, you should see output similar to the following:

.. code-block::

    Launching nvflare/nvflare
    Dashboard will listen to port 443
    /path_to_folder_for_db on host mounted to /var/tmp/nvflare/dashboard in container
    No additional environment variables set to the launched container.
    Dashboard container started
    Container name nvflare-dashboard
    id is 3108eb7be20b92ab3ec3dd7bfa86c2eb83bd441b4da0865d2ebb10bd60612345

We suggest you to set the passphrase to protect the private key of the root CA by using the ``--passphrase`` option.  Once it's set, you have to provide the same passphrase everytime you
restart Dashboard for the same project.

If you would like to start a new project, please remove the db.sqlite file in current working directory (or the directory set with the ``--folder`` option).  Dashboard will start
from scratch and you can provide a project admin email address and get a new password for the project_admin.

The Dashboard will also check the cert folder inside current the working directory (or directory specified by the --folder option) to load web.crt and web.key.
If those files exist, Dashboard will load them and run as an HTTPS server.  If Dashboard does not find both of them, it runs as HTTP server.  In both cases, the service
listens to port 443, unless the ``--port`` option is used to specify a different port. Dashboard will run on ``0.0.0.0``, so by default it should be accessible on the same machine from
``localhost:443``. To make it available to users outside the network, port forwarding and other configurations may be needed to securely direct traffic to the maching running Dashboard.

.. note::

    Running Dashboard requires Docker. You have to ensure your system can pull and run Docker images. The initial docker pull may take some time depending on your network connection.

To stop the running Dashboard, run ``nvflare dashboard --stop``.

**********************************
NVIDIA FLARE Dashboard backend API
**********************************

Architecture
============

The Dashboard backend API follows the Restful concept.  It defines four resources, Project, Organizations, Client and User.  There is one and only one Project.
The Project includes information about server(s) and overseer (if in HA mode).  Clients are defined for NVIDIA FLARE clients and Users for NVIDIA FLARE admin console.
Organizations is a GET only operation, which returns a list of current registered organizations.

Details
=======

API
---
The following is the complete definition of the backend API, written in OpenAPI 3.0 syntax.  Developers can implement the same API in different programming language or
develop different UI while calling the same API for branding purpose.

.. literalinclude:: ../../nvflare/dashboard/dashboard.yaml
  :language: yaml


Authentication and Authorization
--------------------------------
Most of the backend API requires users to login to obtain JWT for authorization purpose.  The JWT includes claims of user's organization and his/her role.  The JWT itself always
has the user's email address (user id for login).

As shown in the above section, only ``GET /project``, ``GET /users`` and ``GET /organizations`` can be called without login credential.

The project_admin role can operate on any resources.


Freezing project
----------------
Because the project itself contains information requires by clients and users, changing project information after clients and users are created will
cause incorrect dependencies.  It is required for the project_admin to freeze the project after all project related information is set and finalized so
that the Dashboard web can allow users to signup.  Once the project is frozen, there is no way, from the Dashboard web, to unfreeze the project.

Database schema
---------------
The following is the schema of the underlying database used by the backend API.

.. image:: ../resources/dashboard_schema.png
    :height: 800px
