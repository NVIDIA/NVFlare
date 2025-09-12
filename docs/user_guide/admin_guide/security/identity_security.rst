#################
Identity Security
#################
This area is concerned with these two trust issues:

    - Authentication: ensures communicating parties have enough confidence about each other's identities: everyone is who they claim to be.
    - Authorization: ensures that the user can only do what he/she is authorized to do.

Authentication
==============
NVFLARE's authentication model is based on Public Key Infrastructure (PKI) technology:

    - For the FL project, the Project Admin uses the Provisioning Tool to create a Root CA with a self-signed root certificate. This Root CA will be used to issue all other certs needed by communicating parties.
    - Identities involved in the study (Server(s), Clients, the Overseer, Users) are provisioned with the Provisioning Tool. Each identity is defined with a unique common name. For each identity, the Provisioning Tool generates a separate password-protected Startup Kit, which includes security credentials for mutual TLS authentication:
        - The certificate of the Root CA
        - The cert of the identity
        - The private key of the identity
    - Startup Kits are distributed to the intended identities:
        - The FL Server's kit is sent to the Project Admin
        - The kit for each FL Client is sent to the Org Admin responsible for the site
        - FLARE Console (previously called Admin Client) kits are sent to the user(s)
    - To ensure the integrity of the Startup Kit, each file in the kit is signed by the Root CA.
    - Each Startup Kit also contains a "start.sh" file, which can be used to properly start the NVFLARE application.
    - Once started, the Client tries to establish a mutually-authenticated TLS connection with the Server, using the PKI credentials in its Startup Kits. This is possible only if the client and the server both have the correct Startup Kits.
    - Similarly, when a user tries to operate the NVFLARE system with the Admin Client app, the admin client tries to establish a mutually-authenticated TLS connection with the Server, using the PKI credentials in its Startup Kits. This is possible only if the admin client and the server both have the correct Startup Kits. The admin user also must enter his/her assigned user name correctly.
 
The security of the system comes from the PKI credentials in the Startup Kits. As you can see, this mechanism involves manual processing and human interactions for Startup Kit distribution, and hence the identity security of the system depends on the trust of the involved people. To minimize security risk, we recommend that people involved follow these best practice guidelines:

    - The Project Admin, who is responsible for the provisioning process of the study, should protect the study's configuration files and store created Startup Kits securely.
    - When distributing Startup Kits, the Project Admin should use trusted communication methods, and never send passwords of the Startup Kits in the same communication. It is preferred to send the Kits and passwords with different communication methods.
    - Org Admin and users must protect their Startup Kits and only use them for intended purposes.
 
.. note::

    The provisioning tool tries to use the strongest cryptography suites possible when generating the PKI credentials. All of the certificates are compliant with the X.509 standard. All private keys are generated with a size of 2048-bits. The backend is openssl 1.1.1f, released on March 31, 2020, with no known CVE.  All certificates expire within 360 days.
 
.. note::

    :ref:`NVFlare Dashboard <nvflare_dashboard_ui>` is a website that supports user and site registration. Users will be able to download their Startup Kits (and other artifacts) from the website.


.. _federated_authorization:

Authorization: Federated Authorization
======================================
Federated learning is conducted over computing resources owned by different organizations. Naturally these organizations have concerns
about their computing resources being misused or abused. Even if an NVFLARE docker is trusted by participating orgs, researchers can
still bring their own custom code to be part of a study (BYOC), which could be a big concern to many organizations. In addition,
organizations may also have IP (intellectual property) requirements on the studies performed by their own researchers.

NVFLARE comes with an authorization system that can help address these security concerns and IP requirements. With this system, an organization can define strict policy to control access to their computing resources and/or FL jobs.

Here are some examples that an org can do:

    - Restrict BYOC to only the org's own researchers;
    - Allow jobs only from its own researchers, or from specified other orgs, or even from specified trusted other researchers;
    - Totally disable remote shell commands on its sites
    - Allow the "ls" shell command but disable all other remote shell commands

Centralized vs. Federated Authorization
---------------------------------------
In NVFLARE before version 2.2.1, the authorization policy was centrally enforced by the FL Server.  In a true federated environment, each organization should be able to define and enforce their own authorization policy instead of relying others (such as FL Server that is owned by a separate org) to do so.

NVFLARE now uses federated authorization where each organization defines and enforces its own authorization policy:

    - Each organization defines its policy in its own authorization.json (in the local folder of the workspace)
    - This locally defined policy is loaded by FL Clients owned by the organization
    - The policy is also enforced by these FL Clients

This decentralized authorization has an added benefit: since each organization takes care of its own authorization, there will be no need to update the policy of any other participants (FL Server or Clients) when a new orgs or clients are added.

See :github_nvflare_link:`Federated Policies (Github) <examples/advanced/federated-policies/README.rst>` for a working example with federated site policies for authorization.

Simplified Authorization Policy Configuration
---------------------------------------------
Since each organization defines its own policy, there will be no need to centrally define all orgs and users. The policy configuration for an org is simply a matrix of role/right permissions. Each role/right combination in the permission matrix answers this question: what kind of users of this role can have this right?

To answer this question, the role/right combination defines one or more conditions, and the user must meet one of these conditions to have the right. The set of conditions is called a control.

Roles
^^^^^
Users are classified into roles. NVFLARE defines four roles:

    - Project Admin - this role is responsible for the whole FL project;
    - Org Admin - this role is responsible for the administration of all sites in its org. Each org must have one Org Admin;
    - Lead (researcher) - this role conducts FL studies
    - Member (researcher) - this role observes the FL study but cannot submit jobs

Rights
^^^^^^
NVFLARE supports more accurate right definitions to be more flexible:

    - Each server-side admin command is a right! This makes it possible for an org to control each command explicitly;
    - Admin commands are grouped into categories. For example, commands like abort_job, delete_job, start_app are in manage_job category; all shell commands are put into the shell_commands category. Each category is also a right.
    - BYOC is now defined as a right so that some users are allowed to submit jobs with BYOC whereas some are not.

This right system makes it easy to write simple policies that only use command categories. It also makes it possible to write policies to control individual commands. When both categories and commands are used, command-based control takes precedence over category-based control.

See :ref:`command_categories` for command categories.

Controls and Conditions
^^^^^^^^^^^^^^^^^^^^^^^
A *control* is a set of one or more conditions that is specified in the permission matrix. Conditions specify relationships among the subject user, the site, and the job submitter. The following are supported relationships:

    - The user belongs to the site's organization (user org = site org)
    - The user is the job submitter (user name = submitter name)
    - The user and the job submitter are in the same org (user org = submitter org)
    - The user is a specified person (user name = specified name)
    - The user is in a specified org (user org = specified org)

Keep in mind that the relationship is always relative to the subject user - we check to see whether the user's name or org has the right relationship with the site or job submitter.

Since conditions need to be expressed in the policy definition file (authorization.json), some concise and consistent notations are needed. The following are the notations for these conditions:

.. csv-table::
    :header: Notation,Condition,Examples
    :widths: 15, 20, 15

    o:site,The user belongs to the site's organization
    n:submitter,The user is the job submitter
    o:submitter,The user and the job submitter belong to the same org
    n:<person_name>,The user is a specified person,n:john@nvidia.com
    o:<org_name>,The user is in a specified org,o:nvidia

The words "site" and "submitter" are reserved.

In addition, two words are used for extreme conditions:

    - Any user is allowed: any
    - No user is allowed: none

See :ref:`sample_auth_policy` for an example policy.

Policy Evaluation
^^^^^^^^^^^^^^^^^
Policy evaluation is to answer the question: is the user allowed to do this command? 

The following is the evaluation algorithm:

    - If a control is defined for this command and user role, then this control will be evaluated;
    - Otherwise, if the command belongs to a category and a control is defined for the category and user role, then this control will be evaluated;
    - Otherwise, return False

As a shorthand, if the control is the same for all rights for a role, you can specify a control for a role without explicitly specifying rights one by one. For example, this is used for the "project_admin" role since this role can do everything.

Command Authorization Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We know that users operate NVFLARE systems with admin commands via the FLARE Console. But when a user issues a command, how does authorization happen
throughout the system?

If the command only involves the Server, then the server's authorization policy is evaluated and
enforced. If the command involves FL clients, then the command will be sent to those clients without any authorization evaluation on the server.
When a client receives the command, it will evaluate its own authorization policy. The client will execute the command only if it passes authorization.
It is therefore possible that some clients accept the command whereas some other clients do not.

If a client rejects the command, it will return "authorization denied" error back to the server.

Job Submission
""""""""""""""
Job submission is a special and important function in NVFLARE. The researcher uses the "submit_job" command to submit a job. But the job
is not executed until it is scheduled and deployed later. Note that when the job is scheduled, the user may or may not be even online.

Job authorization will be done in two places. When the job is submitted, only the Server will evaluate the "submit_job" right. If allowed,
the job will be accepted into the Job Store. When the job is later scheduled for execution, all sites (FL Server and Clients) involved in
the job will evaluate "submit_job" again based on its own authorization policy. If the job comes with custom code, the "byoc" right will
also be evaluated. The job will be rejected if either right fails.

Hence it is quite possible that the job is accepted at submission time, but cannot run due to authorization errors from FL clients.

You may ask why we don't check authorization with each involved FL client at the time of job submission. There are three considerations:

1) This will make the system more complicated since the server would need to interact with the clients
2) At the time of submission, some or all of the FL clients may not even be online
3) A job's clients could be open-ended in that it will be deployed to all available clients. The list of available clients could be different by the time the job is scheduled for execution.

Job Management Commands
"""""""""""""""""""""""
There are multiple commands (clone_job, delete_job, download_job, etc.) in the "manage_jobs" category. Such commands are executed on the Server only and do not involve any FL clients. Hence even if an organization defines controls for these commands, these controls will have no effect.

Job management command authorization often evaluates the relationship between the subject user and the job submitter, as shown in the examples. 

.. _command_categories:

Command Categories
------------------

.. code-block:: python

    class CommandCategory(object):
    
    MANAGE_JOB = "manage_job"
    OPERATE = "operate"
    VIEW = "view"
    SHELL_COMMANDS = "shell_commands"
    
    
    COMMAND_CATEGORIES = {
        AC.ABORT: CommandCategory.MANAGE_JOB,
        AC.ABORT_JOB: CommandCategory.MANAGE_JOB,
        AC.START_APP: CommandCategory.MANAGE_JOB,
        AC.DELETE_JOB: CommandCategory.MANAGE_JOB,
        AC.DELETE_WORKSPACE: CommandCategory.MANAGE_JOB,
        AC.CONFIGURE_JOB_LOG: CommandCategory.MANAGE_JOB,
    
        AC.CHECK_STATUS: CommandCategory.VIEW,
        AC.SHOW_STATS: CommandCategory.VIEW,
        AC.RESET_ERRORS: CommandCategory.VIEW,
        AC.SHOW_ERRORS: CommandCategory.VIEW,
        AC.LIST_JOBS: CommandCategory.VIEW,
    
        AC.SYS_INFO: CommandCategory.OPERATE,
        AC.RESTART: CommandCategory.OPERATE,
        AC.SHUTDOWN: CommandCategory.OPERATE,
        AC.REMOVE_CLIENT: CommandCategory.OPERATE,
        AC.SET_TIMEOUT: CommandCategory.OPERATE,
        AC.CALL: CommandCategory.OPERATE,
        AC.CONFIGURE_SITE_LOG: CommandCategory.OPERATE,
    
        AC.SHELL_CAT: CommandCategory.SHELL_COMMANDS,
        AC.SHELL_GREP: CommandCategory.SHELL_COMMANDS,
        AC.SHELL_HEAD: CommandCategory.SHELL_COMMANDS,
        AC.SHELL_LS: CommandCategory.SHELL_COMMANDS,
        AC.SHELL_PWD: CommandCategory.SHELL_COMMANDS,
        AC.SHELL_TAIL: CommandCategory.SHELL_COMMANDS,
    }


.. _sample_auth_policy:

Sample Policy with Explanations
-------------------------------

This is an example authorization.json (in the local folder of the workspace for a site).

.. code-block:: shell

    {
        "format_version": "1.0",
        "permissions": {
            "project_admin":  "any",   # can do everything on my site
            "org_admin": {
                "submit_job": "none",  # cannot submit jobs to my site
                "manage_job": "o:submitter",  # can only manage jobs submitted by people in the user's own org
                "download_job": "o:submitter", # can only download jobs submitted by people in the user's own org
                "view": "any", # can do commands in the "view" category
                "operate": "o:site",  # can do commands in the "operate" category only if the user is in my org 
                "shell_commands": "o:site"  # can do shell commands only if the user is in my org 
            },
            "lead": {
                "submit_job": "any",  # can submit jobs to my sites
                "byoc": "o:site",  # can submit jobs with BYOC to my sites only if the user is in my org
                "manage_job": "n:submitter", # can only manage the user's own jobs
                "view": "any",  # can do commands in "view" category
                "operate": "o:site", # can do commands in "operate" category only if the user is in my org
                "shell_commands": "none", # cannot do shell commands on my site
                "ls": "o:site",  # can do the "ls" shell command if the user is in my org
                "grep": "o:site"  # can do the "grep" shell command if the user is in my org
            },
            "member": {
                "submit_job": [
                    "o:site",  # can submit jobs to my site if the user is in my org
                    "O:orgA", # can submit jobs to my site if the user is in org "orgA"
                    "N:john" # can submit jobs to my site if the user is "john"
                    ],
                "byoc": "none",  # cannot submit BYOC jobs to my site
                "manage_job": "none",  # cannot manage jobs
                "download_job": "n:submitter",  # can download user's own jobs
                "view": "any",  # can do commands in the "view" category
                "operate": "none"  # cannot do commands in "operate" category
            }
        }
    }

.. _site_specific_auth:

Site-specific Authentication and Federated Job-level Authorization
==================================================================
Site-specific authentication and authorization allows users to inject their own authentication and
authorization methods into the NVFlare system. This includes the FL server / clients registration, authentication,
and the job deployment and run authorization.

NVFlare provides a general purpose event based pluggable authentication and authorization framework to allow for expanding functionality such as:

    - exposing the app through a WAF (Web Application Firewall) or any other network element enforcing Mutual Transport Layer Security(mTLS)
    - using a confidential certification authority to ensure the identity of each participating site and to ensure that they meet the computing requirements for confidential computing
    - defining additional roles to manage who can submit which kind of jobs to execute within NVFlare, identify who submits jobs and which dataset can be accessed

Users can write their own :ref:`FLComponents <fl_component>`, listening to the NVFlare system events at different points of their workflow,
then easily plug in their authentication and authorization logic as needed.

Assumptions and Risks
---------------------
By enabling the customized site-specific authentication and authorization, NVFlare will make several security
related data available to the external FL components, e.g. IDENTITY_NAME, PUBLIC_KEY, CERTIFICATE, etc. In order
to protect them from being compromised, that data needs to be made read-only.

Because of the external pluginable authentication and authorization processes, the results of the processes could
potentially cause the jobs to not be able to be deployed or run. When configuring and using these functions, the users
need to be aware of the impact and know where to plug in the authentication and authorization check.

Event based pluginable authentication and authorization
-------------------------------------------------------
The NVFlare event based solution supports site-specific authentication and federated job-level authorization.
Users can provide and implement any sort of additional security checks by building and plugging in FLcomponents which
listen to the appropriate events and provide custom authentication and authorization functions.

.. code-block:: python

    class EventType(object):
        """Built-in system events."""

        SYSTEM_START = "_system_start"
        SYSTEM_END = "_system_end"
        ABOUT_TO_START_RUN = "_about_to_start_run"
        START_RUN = "_start_run"
        ABOUT_TO_END_RUN = "_about_to_end_run"
        END_RUN = "_end_run"
        SWAP_IN = "_swap_in"
        SWAP_OUT = "_swap_out"
        START_WORKFLOW = "_start_workflow"
        END_WORKFLOW = "_end_workflow"
        ABORT_TASK = "_abort_task"
        FATAL_SYSTEM_ERROR = "_fatal_system_error"
        FATAL_TASK_ERROR = "_fatal_task_error"
        JOB_DEPLOYED = "_job_deployed"
        JOB_STARTED = "_job_started"
        JOB_COMPLETED = "_job_completed"
        JOB_ABORTED = "_job_aborted"
        JOB_CANCELLED = "_job_cancelled"

        BEFORE_PULL_TASK = "_before_pull_task"
        AFTER_PULL_TASK = "_after_pull_task"
        BEFORE_PROCESS_SUBMISSION = "_before_process_submission"
        AFTER_PROCESS_SUBMISSION = "_after_process_submission"

        BEFORE_TASK_DATA_FILTER = "_before_task_data_filter"
        AFTER_TASK_DATA_FILTER = "_after_task_data_filter"
        BEFORE_TASK_RESULT_FILTER = "_before_task_result_filter"
        AFTER_TASK_RESULT_FILTER = "_after_task_result_filter"
        BEFORE_TASK_EXECUTION = "_before_task_execution"
        AFTER_TASK_EXECUTION = "_after_task_execution"
        BEFORE_SEND_TASK_RESULT = "_before_send_task_result"
        AFTER_SEND_TASK_RESULT = "_after_send_task_result"

        CRITICAL_LOG_AVAILABLE = "_critical_log_available"
        ERROR_LOG_AVAILABLE = "_error_log_available"
        EXCEPTION_LOG_AVAILABLE = "_exception_log_available"
        WARNING_LOG_AVAILABLE = "_warning_log_available"
        INFO_LOG_AVAILABLE = "_info_log_available"
        DEBUG_LOG_AVAILABLE = "_debug_log_available"

        PRE_RUN_RESULT_AVAILABLE = "_pre_run_result_available"

        # event types for job scheduling - server side
        BEFORE_CHECK_CLIENT_RESOURCES = "_before_check_client_resources"

        # event types for job scheduling - client side
        BEFORE_CHECK_RESOURCE_MANAGER = "_before_check_resource_manager"

Additional system events
^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    AFTER_CHECK_CLIENT_RESOURCES = "_after_check_client_resources"
    DEPLOY_JOB_TO_SERVER = "_deploy_job_to_server"
    DEPLOY_JOB_TO_CLIENT = "_deploy_job_to_client"

    BEFORE_SEND_ADMIN_COMMAND = "_before_send_admin_command"
    
    BEFORE_CLIENT_REGISTER = "_before_client_register"
    AFTER_CLIENT_REGISTER = "_after_client_register"
    CLIENT_REGISTERED = "_client_registered"
    SYSTEM_BOOTSTRAP = "_system_bootstrap"

    AUTHORIZE_COMMAND_CHECK = "_authorize_command_check"


Security check Inputs
---------------------
Make a ``SECURITY_ITEMS`` dict available in the FLContext, which holds any security check related data.

NVFlare standard data:

.. code-block:: python

    IDENTITY_NAME
    SITE_NAME
    SITE_ORG
    USER_NAME
    USER_ORG
    USER_ROLE
    JOB_META


Security check Outputs
----------------------

.. code-block:: python

    AUTHORIZATION_RESULT
    AUTHORIZATION_REASON

NVFlare will check the ``AUTHORIZATION_RESULT`` to determine if the operations have been authorized to be performed. Before each
operation, the NVFLare platform removes any ``AUTHORIZATION_RESULT`` in the FLContext. After the authorization check process, it
looks for if these results are present in the FLContext or not. If present, it uses its TRUE/FALSE value to determine the action.
If not present, it will be treated as TRUE by default.

Each FLComponent listening and handling the event can use the security data to generate the necessary authorization check
results as needed. The workflow will only continue when all the FLComponents pass the security check. Any one FLComponent
that has the FALSE value will cause the workflow to stop execution.

FLARE Console event support
---------------------------
In order to support additional security data for site-specific customized authentication, we need to add the support for
event based solutions for the FLARE console. Using these events, the FLARE console will be able to add in the custom
SSL certificates, etc, security related data, sent along with the admin commands to the server for site-specific authentication check.

.. code-block:: python

    BEFORE_ADMIN_REGISTER
    AFTER_ADMIN_REGISTER
    BEFORE_SENDING_COMMAND
    AFTER_SENDING_COMMAND
    BEFORE_RECEIVING_ADMIN_RESULT
    AFTER_RECEIVING_ADMIN_RESULT

.. note::

    The site-specific authentication and authorization applies to both FLARE console and :ref:`flare_api`.

Allow more data to be sent to the server for client registration
----------------------------------------------------------------
If the application needs to send additional data from the client to the server to perform the authentication check, the client
can set the data into the FL_Context as public data. Then the server side can get access to the data through the PEER_FL_CONTEXT.
The application can build the FLComponent to listen to the EventType.CLIENT_REGISTERED to perform the authentication check needed.


Site-specific Security Example
------------------------------
To use the site-specific security functions, write a custom Security implementation in the ``local/custom/security_handler.py``,
then configure it as a component in the site ``resources.json``.

.. code-block:: python

    from typing import Tuple

    from nvflare.apis.event_type import EventType
    from nvflare.apis.fl_component import FLComponent
    from nvflare.apis.fl_constant import FLContextKey
    from nvflare.apis.fl_context import FLContext
    from nvflare.apis.job_def import JobMetaKey


    class CustomSecurityHandler(FLComponent):

        def handle_event(self, event_type: str, fl_ctx: FLContext):
            if event_type == EventType.AUTHORIZE_COMMAND_CHECK:
                result, reason = self.authorize(fl_ctx=fl_ctx)
                if not result:
                    fl_ctx.set_prop(FLContextKey.AUTHORIZATION_RESULT, False, sticky=False)
                    fl_ctx.set_prop(FLContextKey.AUTHORIZATION_REASON, reason, sticky=False)

        def authorize(self, fl_ctx: FLContext) -> Tuple[bool, str]:
            command = fl_ctx.get_prop(FLContextKey.COMMAND_NAME)
            if command in ["check_resources"]:
                security_items = fl_ctx.get_prop(FLContextKey.SECURITY_ITEMS)
                job_meta = security_items.get(FLContextKey.JOB_META)
                if job_meta.get(JobMetaKey.JOB_NAME) == "FL Demo Job1":
                    return False, f"Not authorized to execute: {command}"
                else:
                    return True, ""
            else:
                return True, ""

In the ``local/resources.json``:

.. code-block:: json

    {
        "format_version": 2,
        ...
        "components": [
            {
                "id": "resource_manager",
                "path": "nvflare.app_common.resource_managers.gpu_resource_manager.GPUResourceManager",
                "args": {
                "num_of_gpus": 0,
                "mem_per_gpu_in_GiB": 0
                }
            },
            ...
            {
                "id": "security_handler",
                "path": "security_handler.CustomSecurityHandler"
            }
        ]
    }


With the above example, when there is a job named "FL Demo Job1" scheduled to run on this client from the server,
the client will throw the authorization error and prevent the job from running. Any other jobs will be able to execute
on this client.
