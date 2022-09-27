#############
Authorization
#############

NVIDIA FLARE implements a role-based authorization framework that determines what a user can or cannot do based on the user’s
assigned roles configured through AuthPolicyBuilder at provisioning.

********************************
Terminology for FL authorization
********************************
The following concepts are used in defining an authorization policy for NVIDIA FLARE.

Rights
======
A right is a permission for a user to do certain things. For example, the right “train_all” allows the user to do
training for all orgs in a group.

Rules
=====
A rule is a policy that an org wants to enforce. For example, the rule “allow_byoc” allows BYOC code to be included in
the application configurations deployed to the org’s site.

Roles
=====
Even though there may be any number of users, they usually are categorized into several types that share the same
authorization settings. Each such type is called a *role*. A user can be assigned to one or more roles.

Groups
======
Even though there could be many orgs in the study, they usually are categorized into several types that share the same
authorization settings. Each such type is called a *group*. An org can be configured to belong to groups, with a group
for specifying rules for sites of the org and a group for rights definitions.

***************************
Define authorization policy
***************************

Each org can specify its own policies:
    - Orgs that share the same authorization policies are put in the same group, and authorization policies are defined for the group.
    - For each group, the permission matrix is defined for role-right combinations.
    - For each group, permission values are defined for each rule.

The Right Space is a 3D [group, role, right] matrix of permission values, and the Rule Space is a 2D [group, rule] matrix of permission values.

Right Evaluation
================
To determine whether a user has a right on a site:

    - Determine the group(s) that the site belongs to
    - Determine the role(s) of the user
    - Check the Right Space for each [group, role, right] coordinate. If any point is True, then the result is True. This is what we call the “most generous” policy - as long as any of the user roles has the right in any of the groups the site belongs to, the right is granted. If there is no explicit definition for any point, the default value of the right is taken.

.. note::

   Note that what is important is the user’s role(s). The user’s org is not considered except for deciding if a user is
   considered “self” for the site. This can in turn affect the right, for example, if site A is with group configured to
   allow “Operate Self” but not “Operate All” for role “lead_researcher”, a “lead_researcher” user of site A’s org can
   “Operate” on site A whereas a user that is only “lead_researcher” of another org does not have “Operate” rights.

Rule Evaluation
===============
Similar to right evaluation, we also adopt the “most generous” policy to determine the rule value of a site.

Determine the group(s) that the site belongs to
Check the Rule Space for each [group, rule] coordinate. If any point is True, then the result is True.
If there is no explicit definition for any point, the default value of the rule is taken.

Defined Rights
==============
Currently the following rights are defined:

.. csv-table::
   :header: Right,Description

    Upload application,whether the user is allowed to upload applications.
    Deploy All,whether all users of the corresponding role are allowed to deploy applications at sites of a certain group.
    Deploy Self,whether users of the corresponding role and of the same org as sites of a certain group are allowed to deploy applications.
    Train All,whether all users of the corresponding role are allowed to perform training actions at sites of a certain group.
    Train Self,whether users of the corresponding role and of the same org as sites of a certain group are allowed to perform training actions
    View All,whether all users of the corresponding role are allowed to view information at sites of a certain group.
    View Self,whether users of the corresponding role and of the same org as sites of a certain group are allowed to view information
    Operate All,whether all users of the corresponding role are allowed to operate at sites of a certain group.
    Operate Self,whether users of the corresponding role and of the same org as sites of a certain group are allowed to operate

.. note::

   Rights are always in the context of a group, and note that as mentioned above, for evaluating Rights, the rights
   group of the site’s org is what is important, and the user’s org is only important when “All” and “Self” have
   different Rights in that group. Otherwise, the user’s role(s) are all that will matter for determining Rights.

Defined Rules
=============
Currently the following rules are defined:

.. csv-table::
   :header: Rule,Description

    Allow BYOC,whether BYOC code is allowed in application configurations
    Allow Custom Data List,whether custom data list is allowed in application configurations

.. note::

    For these rules to take effect, the user must provide the implementation of the ``TrainConfigValidator``
    and edit the ``config_fed_server.json``, otherwise they take no effect.

*************************
Policy definition example
*************************

The authorization policy is configured in the authz_policy section of study project YAML file: :ref:`project_yml`.  When
using the provisioning tool to generate a set of packages, the authorization policy json file is included in the server's
startup kit zip file.

Here is an example of the generated file::

    {
     "version": "1.0",

     "roles": {
       "super": "super user of system",
       "lead_researcher": "lead researcher of the study",
       "site_researcher": "site researcher of the study",
       "site_it": "site IT of the study",
       "lead_it": "lead IT of the study"
     },
     "groups": {
       "relaxed": {
         "desc": "the org group with relaxed policies",
         "rules": {
           "allow_byoc": true,
           "allow_custom_datalist": true
         }
       },
       "strict": {
         "desc": "the org group with strict policies",
         "rules": {
           "allow_byoc": false,
           "allow_custom_datalist": false
         }
       },
       "general": {
         "desc": "general group user rights",
         "role_rights": {
           "super": {
             "operate_all": true,
             "view_all": true,
             "train_all": true
           },
           "lead_researcher": {
             "train_all": true,
             "view_all": true
           },
           "site_researcher": {
             "train_self": true,
             "view_self": true
           },
           "lead_it": {
             "operate_all": true,
             "view_all": true
           },
           "site_it": {
             "operate_self": true,
             "view_self": true
           }
         }
       }
     },
     "users": {
       "admin@nvidia.com": {
         "org": "nvidia",
         "roles": ["super"]
       },
       "researcher1@org2.com": {
         "org": "org2",
         "roles": ["lead_it", "site_researcher"]
       },
       "researcher2@org1.com": {
         "org": "org1",
         "roles": ["site_researcher"]
       }
     },
     "orgs": {
       "org1": ["general", "strict"],
       "org2": ["general", "relaxed"],
       "nvidia": ["general"]
     },
     "sites": {
       "org1-a": "org1",
       "org1-b": "org1",
       "org2": "org2",
       "server": "nvidia"
     }
    }

A few highlights:

    - Each right has a default value. Default values are used for “holes” in the Right Space.
    - Each rule has a default value. Default values are used for “holes” in the Rule Space.
    - Each user is assigned to a single org and one or more roles;
    - Each site is assigned a single org;
    - Each org is assigned to one or more groups;
    - In each group, a rule and/or right matrix is defined.

****************************
Admin command authorizations
****************************

Each command from the admin user is subject to authorization. The command is executed only if the authorization is passed.

Commands are grouped into the following action groups for rights:

UPLOAD - uploading application configuration to the server.
===========================================================
Command(s) in this group require the “upload application" right.
Furthermore, if the application contains BYOC code, the site’s “allow_byoc” must be true.
Furthermore, if the application contains a custom data list, the site’s “allow_custom_datalist” must be true.

DEPLOY - deploy the application to a site
=========================================
Command(s) in this group require the “deploy all” or “deploy self” right.
Furthermore, if the application contains BYOC code, the site’s “allow_byoc” must be true.
Furthermore, if the application contains a custom data list, the site’s “allow_custom_datalist” must be true.

TRAIN - training related actions (set run, start/abort training)
================================================================
Command(s) in this group require the “train all” or “train self” right.

VIEW - view training and/or system info (ls, head, tail, grep, pwd, …)
================================================================================
Command(s) in this group require the “view all” or “view self” right.

OPERATE - application operation (shutdown, restart server/clients, sys_info)
============================================================================
Command(s) in this group require the “operate all” or “operate self” right.


