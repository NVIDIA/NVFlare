.. _federated_authorization:

#########################
Federated Authorization
#########################

Federated learning is conducted over computing resources owned by different organizations. Naturally these organizations have concerns about their computing resources being misused or abused. Even if an NVFLARE docker is trusted by participating orgs, researchers can still bring their own custom code to be part of a study (BYOC), which could be a big concern to many organizations. In addition, organizations may also have IP (intellectual property) requirements on the studies performed by their own researchers.

NVFLARE comes with an authorization system that can help address these security concerns and IP requirements. With this system, an organization can define strict policy to control access to their computing resources and/or FL jobs.

Here are some examples that an org can do:

    - Restrict BYOC to only the org's own researchers;
    - Allow jobs only from its own researchers, or from specified other orgs, or even from specified trusted other researchers;
    - Totally disable remote shell commands on its sites
    - Allow the "ls" shell command but disable all other remote shell commands

Centralized vs. Federated Authorization
========================================
In NVFLARE before version 2.2.1, the authorization policy was centrally enforced by the FL Server.  In a true federated environment, each organization should be able to define and enforce their own authorization policy instead of relying others (such as FL Server that is owned by a separate org) to do so.

NVFLARE 2.2.1 changes the way authorization is implemented to federated authorization where each organization defines and enforces its own authorization policy:

    - Each organization defines its policy in its own authorization.json (in the local folder of the workspace)
    - This locally defined policy is loaded by FL Clients owned by the organization
    - The policy is also enforced by these FL Clients

This decentralized authorization has an added benefit: since each organization takes care of its own authorization, there will be no need to update the policy of any other participants (FL Server or Clients) when a new orgs or clients are added.

See `Federated Policies (Github) <https://github.com/NVIDIA/NVFlare/blob/main/examples/advanced/federated-policies/README.rst>`_ for a working example with federated site policies for authorization.

Simplified Authorization Policy Configuration
==============================================
Since each organization defines its own policy, there will be no need to centrally define all orgs and users. The policy configuration for an org is simply a matrix of role/right permissions. Each role/right combination in the permission matrix answers this question: what kind of users of this role can have this right?

To answer this question, the role/right combination defines one or more conditions, and the user must meet one of these conditions to have the right. The set of conditions is called a control.

Roles
-----
Users are classified into roles. NVFLARE defines four roles starting in 2.2.1:

    - Project Admin - this role is responsible for the whole FL project;
    - Org Admin - this role is responsible for the administration of all sites in its org. Each org must have one Org Admin;
    - Lead (researcher) - this role conducts FL studies
    - Member (researcher) - this role observes the FL study but cannot submit jobs

Rights
------
NVFLARE 2.2.1 supports more accurate right definitions to be more flexible:

    - Each server-side admin command is a right! This makes it possible for an org to control each command explicitly;
    - Admin commands are grouped into categories. For example, commands like abort_job, delete_job, start_app are in manage_job category; all shell commands are put into the shell_commands category. Each category is also a right.
    - BYOC is now defined as a right so that some users are allowed to submit jobs with BYOC whereas some are not.

This right system makes it easy to write simple policies that only use command categories. It also makes it possible to write policies to control individual commands. When both categories and commands are used, command-based control takes precedence over category-based control.

See :ref:`command_categories` for command categories.

Controls and Conditions
-----------------------
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
-----------------
Policy evaluation is to answer the question: is the user allowed to do this command? 

The following is the evaluation algorithm:

    - If a control is defined for this command and user role, then this control will be evaluated;
    - Otherwise, if the command belongs to a category and a control is defined for the category and user role, then this control will be evaluated;
    - Otherwise, return False

As a shorthand, if the control is the same for all rights for a role, you can specify a control for a role without explicitly specifying rights one by one. For example, this is used for the "project_admin" role since this role can do everything.

Command Authorization Process
-----------------------------
We know that users operate NVFLARE systems with admin commands via the FLARE Console. But when a user issues a command, how does authorization happen throughout the system? In NVFLARE 2.1 and before, the authorization policy is evaluated and enforced by the FL Server that processes the command. But in NVFLARE 2.2, this is totally changed.

The command is still received by the FL Server. If the command only involves the Server, then the server's authorization policy is evaluated and enforced. If the command involves FL clients, then the command will be sent to those clients without any authorization evaluation on the server. When a client receives the command, it will evaluate its own authorization policy. The client will execute the command only if it passes authorization. It is therefore possible that some clients accept the command whereas some other clients do not.

If a client rejects the command, it will return "authorization denied" error back to the server.

Job Submission
^^^^^^^^^^^^^^
Job submission is a special and important function in NVFLARE. The researcher uses the "submit_job" command to submit a job. But the job is not executed until it is scheduled and deployed later. Note that when the job is scheduled, the user may or may not be even online.

Job authorization will be done in two places. When the job is submitted, only the Server will evaluate the "submit_job" right. If allowed, the job will be accepted into the Job Store. When the job is later scheduled for execution, all sites (FL Server and Clients) involved in the job will evaluate "submit_job" again based on its own authorization policy. If the job comes with custom code, the "byoc" right will also be evaluated. The job will be rejected if either right fails.

Hence it is quite possible that the job is accepted at submission time, but cannot run due to authorization errors from FL clients.

You may ask why we don't check authorization with each involved FL client at the time of job submission. There are three considerations:

1) This will make the system more complicated since the server would need to interact with the clients
2) At the time of submission, some or all of the FL clients may not even be online
3) A job's clients could be open-ended in that it will be deployed to all available clients. The list of available clients could be different by the time the job is scheduled for execution.

Job Management Commands
^^^^^^^^^^^^^^^^^^^^^^^
There are multiple commands (clone_job, delete_job, download_job, etc.) in the "manage_jobs" category. Such commands are executed on the Server only and do not involve any FL clients. Hence even if an organization defines controls for these commands, these controls will have no effect.

Job management command authorization often evaluates the relationship between the subject user and the job submitter, as shown in the examples. 

.. _command_categories:

Appendix One - Command Categories
=================================

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
    
        AC.SHELL_CAT: CommandCategory.SHELL_COMMANDS,
        AC.SHELL_GREP: CommandCategory.SHELL_COMMANDS,
        AC.SHELL_HEAD: CommandCategory.SHELL_COMMANDS,
        AC.SHELL_LS: CommandCategory.SHELL_COMMANDS,
        AC.SHELL_PWD: CommandCategory.SHELL_COMMANDS,
        AC.SHELL_TAIL: CommandCategory.SHELL_COMMANDS,
    }


.. _sample_auth_policy:

Appendix Two - Sample Policy with Explanations
==============================================

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
