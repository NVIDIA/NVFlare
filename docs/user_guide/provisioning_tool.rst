.. _provisioning:

############################
Provisioning in NVIDIA FLARE
############################
A necessary first step in establishing a federation is provisioning to establish the identities of the server, clients,
and admin clients.

When operating federated learning, communication channels use shared SSL certificates generated
during provisioning to establish the identities and secure communication between participants.

Provisioning in NVIDIA FLARE generates mutual-trusted system-wide configurations for all participants
so all of them can join the NVIDIA FLARE system across different locations. To achieve this, a provisioning tool powered
by the Open Provision API and its builder modules is included in NVIDIA FLARE (:mod:`nvflare.lighter`)
to create a startup kit for each participant with the necessary configuration artifacts.

The configurations usually include, but are not limited to, the following information:

    - network discovery, such as domain names, port numbers or IP addresses
    - credentials for authentication, such as certificates of participants and root authority
    - authorization policy, such as roles, rights and rules
    - tamper-proof mechanism, such as signatures
    - convenient commands, such as shell scripts with default command line options to easily start an individual participant

In order to enable developers to freely add / modify / remove the above configurations to fit their own requirements,
we created the Open Provision API. Now developers can take advantage of this API to perform the provisioning tasks
which meet their own requirements in addition to the default provisioning before for creating packages for the
server, client, and administrators.

*******************************
NVIDIA FLARE Open Provision API
*******************************

Architecture
============

.. image:: ../resources/Open_Provision_API.png
    :height: 350px

The above diagram describes the architecture of NVIDIA FLARE Open Provision API in blue. Those two green blocks are the
sample python code (provision.py) collecting project configuration information (project.yml) and interacting with
components of Open Provision API to accomplish a provisioning task. The Provisioner and blocks inside the blue box are
classes or subclasses of Open Provision API.

Details
=======

project.yml
-----------
This is a simple yaml file, describing participants and builders.  Note that Open Provision API itself does not define
the format of this file.  Any developer can create his/her own file that describes participants and builders in a
different format.  The developer can even store such information in one URL as long as there is code
(provision.py in the above sample diagram) that can load the information and convert such information to calls
to Open Provision API.

provision.py
------------
This python file is the sample application to interact with the Open Provision API.  It also loads project.yml, parses
command line options, instantiates classes/subclasses defined from Open Provision API and displays helpful messages to users.
As mentioned previously, developers are encouraged to modify provision.py or write their own applications that fit their own requirements.
It is also possible to completely utilize Open Provision API without any standalone applications.  For example, if
developers have their existing applications and would like to add provisioning capabilities for the NVIDIA FLARE system,
they can add API calls to Open Provision API to generate required outputs.

Provisioner
-----------
This is the container class that owns all instances of Project, Workspace, Provision Context, Builders and Participants,
as shown in the above diagram.  A typical usage of this class is like the following::

    provisioner = Provisioner(workspace_full_path, builders)

    provisioner.provision(project)

Project
-------
The Project class keeps information about participants.  Therefore, information of any participant can be retrieved from
the Project instance::

    class Project(object):
       def __init__(self, name: str, description: str, participants: List[Participant]):
           self.name = name
           all_names = list()
           for p in participants:
               if p.name in all_names:
                   raise ValueError(f"Unable to add a duplicate name {p.name} into this project.")
               else:
                   all_names.append(p.name)
           self.description = description
           self.participants = participants

       def get_participants_by_type(self, type, first_only=True):
           found = list()
           for p in self.participants:
               if p.type == type:
                   if first_only:
                       return p
                   else:
                       found.append(p)
           return found

Participant
-----------
Each participant is one entity that communicates with other participants inside the NVIDIA FLARE system during runtime.
Each participant has the following attributes: type, name, org and props.  The attribute ``props`` is a dictionary and
stores additional information::

    class Participant(object):
       def __init__(self, type: str, name: str, org: str, *args, **kwargs):
           self.type = type
           self.name = name
           self.org = org
           self.subject = name
           self.props = kwargs

The name of each participant must be unique.  This is enforced in Project's __init__ method.  The type
defines the behavior of this participant when it is alive in the NVIDIA FLARE system.  For example, type = 'server' defines
that the participant acts as a server.  Three types are commonly used for a typical NVIDIA FLARE system: server, client, and
admin.  However, developers can freely add other types when needed, such as 'gateway,' 'proxy,' or 'database.'  The
builders can take such information into consideration so that they can generate relevant results based on the type
attribute.

Builder
-------
The builders in the above diagram are provided as a convenient way to generate commonly used zip files for a typical
NVIDIA FLARE system.  Developers are encouraged to add / modify or even remove those builders to fit their own requirements.

Each builder is responsible for taking the information from project, its own __init__ arguments, and provisioner to
generate data.  For example, the HEBuilder is responsible for generating tenseal context files for server and client,
but not admin.  Additionally, the context for servers does not include either public key or secret key while the
context for clients include both.  Its __init__ arguments consist of poly_modules_degree, coeff_mod_bit_sizes,
scale_bits and scheme.  With all of the information, HEBuilder can output context files correctly.

Provisioner calls each builder's initialize method first during provisioning time in a loop.  This allows builders to
prepare information and to populate their instance variables.  After calling each builder's initialize method, the
Provisioner calls each builder's build method in another loop.  This method is usually implemented to execute the
actual build process (generating necessary files).  At the end, the provisioner calls the finalize method of each
builder in REVERSE ORDER in the third loop so all builders have a chance to wrap up their states.  This comes from
the convention that the earlier a builder's initialize method is called, the later its finalize method should be called.

The iterations in each of the above three loops are always determined by the builders list, the second argument
passed to Provisioner class.  Therefore, different orders in the builders list affect the results.

For example, when one builder's finalize method cleans up and removes the wip folder which is shared by all builders,
builders being called after it will not be able to access the wip folder.

.. note:: The collaboration among all builders is the responsibility of Open Provision API developers.

Every builder has to subclass the Builder class and override one or more of these three methods::

    class Builder(ABC):
       def initialize(self, ctx: dict):
           pass

       def build(self, project: Project, ctx: dict):
           pass

       def finalize(self, ctx: dict):
           pass

Workspace
---------
Each builder can access four folders under provision workspace which is managed by Provisioner (see Provisioner's
first argument).  Those folders are 'wip' (for working-in-progress), 'kit_dir' (a subfolder in 'wip'), 'state' (used
to persist information between different revisions) and 'resources' (for read-only / static information).

Provision Context
-----------------
Provision context is created by Provisioner and can be read / written by all participants and builders.  A builder
might add a piece of information to it so that another builder can retrieve it.  As a hypothetical example, developers
might want to add a second homomorphic encryption builder to generate a different set of HE contexts based on
certificates from CertBuilder and the context of the first HE builder.  To achieve this, the developers can write
certificates to provision context at CertBuilder and he context to provision context at HEBuilder.  The information
is automatically available to the second HE builder.

Open Provision API Case Studies
===============================
Before we start, please remember that the builders have three methods to be implemented optionally, initialize, build
and finalize.  The Provisioner calls initialize methods of all builders, then build methods of all builders.  Both in
the order of builders list.  However, the finalize methods of all builders are called by Provisioner in REVERSE order.
Please keep this in mind.

For example, in Case 2, the builders is a list and append method adds the WebPostDistributionBuilder to the end of the
builder list.  As mentioned above, the initialize and build methods are called in the order of the builder list while
the finalize method is called in the reverse order.  We can expect the finalize method of the WebPostDistributionBuilder
is called before other builders' finalize methods and before other builders' build methods.

Case 1: generating additional files
-----------------------------------
The developers would like to add a configuration file about a database server to admin participants.  The configuration
is like this::

    [database]
    db_server = server name
    db_port = port_number
    user_name = admin's name

As this requires adding one file to every admin participant, the developer can write a DBBuilder as follows::

    class DBConfigBuilder(Builder):
       def __init__(self, db_server, db_port):
           self.db_server = db_server
           self.db_port = db_port

       def build(self, project, ctx):
           for admin in project.get_participants_by_type("admin", first_only=False):
               dest_dir = self.get_kit_dir(admin, ctx)
               with open(os.path.join(dest_dir, "database.conf"), 'wt') as f:
                   f.write("[database]\n")
                   f.write(f"db_server = {self.db_server}\n")
                   f.write(f"db_port = {self.db_port}\n")
                   f.write(f"user_name = {admin.name}\n")

And in project.yml, add an entry in the builders section::

    - path: byob.DBConfigBuilder
      args:
        db_server: example.com
        db_port: 5432

Case 2: enhancing an existing builder
-------------------------------------
The developer would like to push the generated zip file, done by nvflare.lighter.impl.workspace.DistributionBuilder, to
a web server via a POST method.  This can be done easily by replacing the DistributionBuilder with a new builder as
follows (after pip install requests)::

    class WebPostDistributionBuilder(Builder):
       def __init__(self, url):
           self.url = url

       def build(self, project: Project, ctx: dict):
           wip_dir = self.get_wip_dir(ctx)
           dirs = [name for name in os.listdir(wip_dir) if os.path.isdir(os.path.join(wip_dir, name))]
           for dir in dirs:
               dest_zip_file = os.path.join(wip_dir, f"{dir}")
               shutil.make_archive(dest_zip_file, "zip", root_dir=os.path.join(wip_dir, dir), base_dir="startup")
               files = {"upload_file": open(dest_zip_file, "rb")}
               r = requests.post(self.url, files=files)

And just replace the existing one with the new builder under Builders in the project.yml::

    - path: byob.WebPostDistributionBuilder
      args:
        url: https://example.com/nvflare/provision

For the above two cases, if developers opt to use Open Provision API directly instead of project.yml, they can do this
(some code omitted for clarity)::

    from byob import WebPostDistributionBuilder
    builders = list()
    # Adding other builders
    # ...

    # Using our new WebPostDistributionBuilder builders.append(WebPostDistributionBuilder(url="https://example.com/nvflare/provision"))

    # Instantiate Provisioner
    provisioner = Provisioner(workspace_full_path, builders)

Case 3: adding both new builders and participants of new types
--------------------------------------------------------------
The developers would like to add participants of type = 'gateway.'  In order to handle this type of participants, a new
builder is needed to write gateway specific configuration.  First, specify that in project.yml::

    - name: gateway1
      type: gateway
      org: nvidia
      port: 8102

or in API style::

    participants = list()
    p = Participant(name="gateway1", type="gateway", org="nvidia", port=8102)
    participants.append(p)

A new builder to write 'gateway.conf' can be implemented as follows (for reference)::

    class GWConfigBuilder(Builder):
      def build(self, project, ctx):
          for gw in project.get_participants_by_type("gateway", first_only=False):
              dest_dir = self.get_kit_dir(gw, ctx)
              with open(os.path.join(dest_dir, "gateway.conf"), 'wt') as f:
                  port = gw.props.get("port")
                  f.write("[gateway]\n")
                  f.write(f"name = {gw.name}\n")
                  f.write(f"port = {port}\n")

Takeaways for Custom Builders
-----------------------------
From the cases shown previously, implementing your own Builders only requires the following steps:

#. Subclass the Builder class
#. Implement the required methods (initialize, build, finalize).  Not all of them have to be implemented.
#. The builder can locate the working-in-progress space from the return value of this method self.get_wip_dir(ctx).  This
   space is shared by all builders.
#. Builder writes participant-specific files to the kit directory which is the return value of self.get_kit_dir(participant, ctx)
#. Builders have to coordinate with one another.  For example, the WebPostDistributionBuilder generates zip files from the
   contents inside kit directories.  That implies some other builders have to write those contents first.

.. _bundled_builders:

Bundled builders
================
The following is the list of bundled builders included by default in the NVIDIA FLARE package.  They are provided as a
convenient tool.  As mentioned previously, developers are encouraged to add / modify / remove builders based on their
own requirements:

    - :class:`WorkspaceBuilder<nvflare.lighter.impl.workspace.WorkspaceBuilder>`
    - :class:`TemplateBuilder<nvflare.lighter.impl.template.TemplateBuilder>`
    - :class:`StaticFileBuilder<nvflare.lighter.impl.static_file.StaticFileBuilder>`
    - :class:`AuthPolicyBuilder<nvflare.lighter.impl.auth_policy.AuthPolicyBuilder>`
    - :class:`CertBuilder<nvflare.lighter.impl.cert.CertBuilder>`
    - :class:`HEBuilder<nvflare.lighter.impl.he.HEBuilder>`
    - :class:`SignatureBuilder<nvflare.lighter.impl.signature.SignatureBuilder>`
    - :class:`DistributionBuilder<nvflare.lighter.impl.workspace.DistributionBuilder>`

::

    workspace structure
    └── example_project
        ├── prod_00
        │   ├── admin@nvidia.com
        │   │   └── startup
        │   ├── localhost
        │   │   └── startup
        │   ├── site1
        │   │   └── startup
        │   └── site2
        │       └── startup
        ├── prod_01
        │   ├── admin@nvidia.com
        │   │   └── startup
        │   ├── localhost
        │   │   └── startup
        │   ├── site1
        │   │   └── startup
        │   └── site2
        │       └── startup
        ├── prod_02
        │   ├── admin@nvidia.com
        │   │   └── startup
        │   ├── localhost
        │   │   └── startup
        │   ├── site1
        │   │   └── startup
        │   └── site2
        │       └── startup
        ├── resources
        └── state

The prod_NN folders contain the provisioning results.  The number, NN, increases every time the provision command runs successfully.

*****************
Project yaml file
*****************

This is the key file that describes the information which provisioning tool will be using to generate startup kits for server, clients and admins.
If there is no ``project.yml`` in your current working directory, simply run ``provision`` without any option.  It
will ask you if you would like to have one sample copy of this file created.

.. code-block:: shell

  (nvflare-venv) ~/workspace$ provision
  No project.yml found in current folder.  Is it OK to generate one at /home/nvflare/workspace/project.yml for you? (y/N) y
  /home/nvflare/workspace/project.yml was created.  Please edit it to fit your FL configuration.


Edit the project.yml configuration file to meet your project requirements:

    - "api_version" must be 3 for current release of provisioning tool
    - "name" is used to identify this project.
    - "participants" describes the different parties in the FL system, distinguished by type. For all participants, "name"
      should be unique, and "org" should be defined in AuthPolicyBuilder. The "name" of the Overseer and servers should
      be in the format of fully qualified domain names. It is possible to use a unique hostname rather than FQDN, with
      the IP mapped to the hostname by having it added to ``/etc/hosts``.
        - Type "overseer" describes the Overseer, with the "org", "name", "protocol", "api_root", and "port".
        - Type "server" describes the FL servers, with the "org", "name", "fed_learn_port", "admin_port", and "enable_byoc".
            - "fed_learn_port" is the port number for communication between the FL server and FL clients
            - "admin_port" is the port number for communication between the FL server and FL administration client
        - Type "client" describes the FL clients, with one "org" and "name" for each client as well as "enable_byoc" settings.
        - Type "admin" describes the admin clients with the name being a unique email. The roles must be defined in AuthPolicyBuilder below.
    - "builders" contains all of the builders and the args to be passed into each. See the details in docstrings of the :ref:`bundled_builders`.
    - See :ref:`system_components` for information on the components configured in StaticFileBuilder.

.. note::

   For each participant, the ``enable_byoc`` flag can be set to enable loading of code in the custom folder of applications.
   If the ``enable_byoc`` flag is disabled, even if you have custom code in your application folder, it will not be loaded.

   There is also a setting for ``allow_byoc`` in the rules for authorization groups (in AuthPolicyBuilder). This controls
   whether or not applications containing custom code will be allowed to be uploaded and deployed to the participants
   of the orgs of that rule group.

   Here, ``byoc`` is referring to the custom code in the custom folder in an FL application. Code already in the python path
   through other means is not considered ``byoc`` for these purposes.

.. _project_yml:

Default project.yml file
========================

The following is an example of the default project.yml file.

.. literalinclude:: ../../nvflare/lighter/project.yml
  :language: yaml

.. attention:: Please make sure that the Overseer and FL servers ports are accessible by all participating sites.

*****************************
Provision commandline options
*****************************

Running ``provision -h`` shows all available options.

.. code-block:: shell

  (nvflare_venv) ~/workspace/repos/flare$ provision -h
  usage: provision [-h] [-p PROJECT_FILE] [-w WORKSPACE] [-c CUSTOM_FOLDER] [-u]

  optional arguments:
    -h, --help                                               show this help message and exit
    -p PROJECT_FILE, --project_file PROJECT_FILE                 file to describe FL project
    -w WORKSPACE, --workspace WORKSPACE                          directory used by provision
    -c CUSTOM_FOLDER, --custom_folder CUSTOM_FOLDER    additional folder to load python code
    -u, --ui_tool                      Run provisioning UI tool to generate project.yml file

Running ``provision`` without any options and without a project.yml file in the current working directory will prompt
to copy a default project.yml to the current working directory.

*************************
Provisioning tool UI page
*************************

The ``-u`` option will open the provisioning tool helper UI page in your browser.  This tool is built to help with
setting up and generating a project.yml to work with the reference configuration of default bundled builders in the
NVIDIA FLARE package. You may need to add to or edit the builders section before running ``provision -p project.yml`` if you
have customized builders.

.. image:: ../resources/provisioning_ui.png
    :height: 400px
