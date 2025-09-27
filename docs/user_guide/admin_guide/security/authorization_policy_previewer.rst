.. _authorization_policy_previewer:

******************************
Authorization Policy Previewer
******************************

:ref:`Authorization <federated_authorization>` is an important security feature of NVFLARE. Since NVFLARE 2.2, each site defines its own authorization policy.
Since authorization policy is vital for system security, and many people can now define policies, it's important to be able
to validate the policies before deploying them to production.

The Authorization Policy Previewer is a tool for validating authorization policy definitions. The tool provides an interactive
user interface and commands for users to validate different aspects of policy definitions:

    - Show defined roles and rights
    - Show the content of the policy definition
    - Show the permission matrix (role/right/conditions)
    - Evaluate a right against a specified user

Start Authorization Policy Previewer
======================================
To start the Authorization Policy Previewer, enter this command on a terminal:

.. code-block:: shell

  nvflare authz_preview -p <authorization_policy_file>

The authorization_policy_file must be a JSON file that follows authorization file format.

If the file is not a valid JSON file or does not follow authorization file format, this command will exit with exception.

Execute Authorization Policy Previewer Commands
================================================
If the Authorization Policy Previewer is successfully started, the prompt ``>`` will be displayed and for command input.

To get the complete list of commands, enter "?" on the prompt.

Most commands are self-explanatory, except for the "eval_right". With this command, you can evaluate a specified right against a
specified user (name:org:role) to make sure the result is correct.

Role Rights
===========
Most permissions in the policy file may be defined with Command Categories. However, once the policy file is loaded, categories are
already resolved to individual commands, following the fallback mechanism.

Use the ``show_role_rights command`` to verify that all commands have the right permissions for all roles.

Evaluate a Right
================
The syntax of the ``eval_right`` command is:

.. code-block:: shell

  eval_right site_org right_name user_name:org:role [submitter_name:org:role]

where:

.. code-block::

    site_org - the organization of the site
    right_name - the right to be evaluated. You can use the "show_rights" command to list all available commands.
    User specification - a user spec has three pieces of information separated by colons. Name is the name of the user; org is the organization that the user belongs to; and role is the user's role. You can use the "show_roles" command to list all available roles.
    Submitter specification - some job related commands can evaluate the relation between the user and the submitter of a job. Submitter spec has the same format as user spec.

Please refer to :ref:`Federated Authorization <federated_authorization>` for details on the right definition and evaluation.

Stop Authorization Policy Previewer
======================================
To exit from the Authorization Policy Previewer, enter the "bye" command at the prompt.
