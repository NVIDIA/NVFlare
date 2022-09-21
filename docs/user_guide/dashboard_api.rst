.. _dashboard_api:

#########################
Dashboard in NVIDIA FLARE
#########################
As mentioned in :ref:`provisioning`, NVIDIA FLARE system requires a set of startup kits
which include the private keys and certificates, signed by the root CA, in order to communicate to one another.
The new Dashboard in NVIDIA FLARE provides a simple way to collect information of clients and users from different organizations,
as well as to generate those startup kits for users to download.
 
Most of the details about provisioning can be found in :ref:`provisioning`.  In this section, we focus on the user interaction with Dashboard and its backend API.


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
    -i DASHBOARD_IMAGE, --dashboard_image DASHBOARD_IMAGE
                            container image for running dashboard
    --passphrase PASSPHRASE
                            Passphrase to encrypt/decrypt root CA private key. !!! Do not share it with others. !!!
    -e ENV, --env ENV     additonal environment variables: var1=value1

To start Dashboard, run ``nvflare dashboard --start``.  For the first time, it may take a while to download the nvflare image.

We suggest you to set the passphrase to protect the private key of root CA.  Once it's set, you have to provide the same passphrase everytime you
restart the dashboard for the same project.

Dashboard docker will detect if the database is initialized.  If not, it will ask the project_admin email address and will generate a random password.  Please
login with this credential as project_admin to finish the project setting in Dashboard.  The project_admin can change his/her password in the Dashboard.

If you would like to start a new project, please remove the db.sqlite file in current working directory (or the directory set in --folder option).  Dashboard will start
from scratch and proceed as the previous paragraph.

The Dashboard will also check the cert folder inside current working directory (or --folder option) to load web.crt and web.key.  If those files exists, Dashboard will
load them and run as HTTPS server.  If Dashboard does not find both of them, it runs as HTTP server.  In both cases, the service listens to port 443, unless it's set otherwise
by --port option.


.. note::

    Running Dashboard requires docker.  You have to ensure your system can pull and run docker images

To stop the running Dashboard, run ``nvflare dashboard --stop``.
