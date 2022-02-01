# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import List


def _validate_value(type_def: dict, value):
    value_type = type_def.get("type", "bool")
    if value_type == "bool":
        if value not in [True, False]:
            return False
    elif not isinstance(value, (int, float)):
        return False
    return True


def validate_policy_config(config: dict) -> str:
    """Validates that an authorization policy configuration has the right syntax.

    Args:
        config: configuration dictionary to validate

    Returns: empty string if there are no errors, else a string describing the error encountered

    """
    if not isinstance(config, dict):
        return "policy definition must be a dict"

    # validate rules
    policy_rules = config.get("rules", None)
    if not policy_rules:
        return "missing rules in policy"
    if not isinstance(policy_rules, dict):
        return "rules must be dict"
    for rule_name, rule_def in policy_rules.items():
        if not isinstance(rule_def, dict):
            return 'bad rule "{}": must be a dict'.format(rule_name)

        default_value = rule_def.get("default", None)
        if default_value is None:
            return 'bad rule "{}": missing default value'.format(rule_name)

        rule_type = rule_def.get("type", "bool")
        if rule_type not in ["bool", "int"]:
            return 'bad rule "{}": invalid type "{}"'.format(rule_name, rule_type)

        if not _validate_value(rule_def, default_value):
            return 'bad rule "{}": invalid default "{}"'.format(rule_name, default_value)

    # validate rights
    policy_rights = config.get("rights", None)
    if not policy_rights:
        return "missing rights in policy"
    if not isinstance(policy_rights, dict):
        return "rights must be dict"
    for right_name, right_def in policy_rights.items():
        if not isinstance(right_def, dict):
            return 'bad right "{}": must be a dict'.format(right_name)

        precond = right_def.get("precond", None)
        if precond is not None and precond not in ["selfOrg"]:
            return 'bad right "{}": unknown precond "{}"'.format(right_name, precond)

        default_value = right_def.get("default", None)
        if default_value is None:
            return 'bad right "{}": missing default value'.format(right_name)

        right_type = right_def.get("type", "bool")
        if right_type not in ["bool", "int"]:
            return 'bad right "{}": invalid type "{}"'.format(right_name, right_type)

        if not _validate_value(right_def, default_value):
            return 'bad right "{}": invalid default "{}"'.format(right_name, default_value)

    # validate roles
    policy_roles = config.get("roles", {})
    if policy_roles:
        if not isinstance(policy_roles, dict):
            return "roles must be dict"

        for role_name, role_desc in policy_roles.items():
            if not isinstance(role_name, str):
                return 'bad role name "{}": must be str'.format(role_name)
            if not isinstance(role_desc, str):
                return 'bad role desc "{}": must be str'.format(role_desc)

    # validate groups
    policy_groups = config.get("groups", {})
    if policy_groups:
        if not isinstance(policy_groups, dict):
            return "groups must be dict"

        for grp_name, grp_def in policy_groups.items():
            if not isinstance(grp_def, dict):
                return 'bad group "{}": group def must be dict'.format(grp_name)
            rules = grp_def.get("rules", None)
            if rules:
                if not isinstance(rules, dict):
                    return 'bad group "{}": rules must be dict'.format(grp_name)
                for rule_name, rule_value in rules.items():
                    rule_def = policy_rules.get(rule_name, None)
                    if not rule_def:
                        return 'bad group "{}": unknown rule "{}"'.format(grp_name, rule_name)
                    if not _validate_value(rule_def, rule_value):
                        return 'bad group "{}": invalid value "{}" for rule "{}"'.format(
                            grp_name, rule_value, rule_name
                        )

            role_rights = grp_def.get("role_rights", None)
            if role_rights:
                if not isinstance(role_rights, dict):
                    return 'bad group "{}": role_rights must be dict'.format(grp_name)
                for role_name, rights in role_rights.items():
                    if not isinstance(rights, dict):
                        return 'bad group "{}": rights of role "{}" must be dict'.format(grp_name, role_name)
                    for right_name, right_value in rights.items():
                        right_def = policy_rights.get(right_name, None)
                        if not right_def:
                            return 'bad group "{}": unknown right "{}" in role "{}" must be dict'.format(
                                grp_name, right_name, role_name
                            )
                        if not _validate_value(right_def, right_value):
                            return 'bad group "{}": invalid value "{}" for right "{}" in role "{}" must be dict'.format(
                                grp_name, right_value, right_name, role_name
                            )

    # validate orgs
    policy_orgs = config.get("orgs", None)
    if not policy_orgs:
        return "missing orgs in policy"

    if not isinstance(policy_orgs, dict):
        return "orgs must be a dict"

    for org_name, groups in policy_orgs.items():
        if not isinstance(groups, list):
            return 'bad org "{}": groups must be a list'.format(org_name)
        if len(groups) <= 0:
            return 'bad org "{}": groups not defined'.format(org_name)
        for grp_name in groups:
            if not isinstance(grp_name, str):
                return 'bad org "{}": group name must be str'.format(org_name)
            grp_def = policy_groups.get(grp_name, None)
            if not grp_def:
                return 'bad org "{}": undefined group "{}"'.format(org_name, grp_name)

    # validate users
    policy_users = config.get("users", None)
    if not policy_users:
        return "missing users in policy"

    if not isinstance(policy_users, dict):
        return "users must be dict"

    for user_name, user_def in policy_users.items():
        if not isinstance(user_def, dict):
            return 'bad user "{}": definition must be dict'.format(user_name)
        org_name = user_def.get("org", None)
        if not org_name:
            return 'bad user "{}": missing org'.format(user_name)

        org_def = policy_orgs.get(org_name)
        if not org_def:
            return 'bad user "{}": undefined org "{}"'.format(user_name, org_name)

        roles = user_def.get("roles", None)
        if not roles:
            return 'bad user "{}": missing roles'.format(user_name)
        if not isinstance(roles, list):
            return 'bad user "{}": roles must be list'.format(user_name)
        if len(roles) <= 0:
            return 'bad user "{}": no roles defined'.format(user_name)
        for role_name in roles:
            if not isinstance(role_name, str):
                return 'bad user "{}": role name must be str'.format(user_name)
            role_def = policy_roles.get(role_name, None)
            if not role_def:
                return 'bad user "{}": undefined role "{}" must be str'.format(user_name, role_name)

    # validate sites
    sites = config.get("sites", None)
    if not sites:
        return "missing sites in policy"

    if not isinstance(sites, dict):
        return "sites must be dict"

    for site_name, org_name in sites.items():
        if org_name not in policy_orgs:
            return 'bad site "{}": undefined org "{}"'.format(site_name, org_name)

    return ""


def _group_rule_key(grp_name: str, rule_name: str):
    return grp_name + ":" + rule_name


def _group_role_right_key(grp_name: str, role_name: str, right_name: str):
    return grp_name + ":" + role_name + ":" + right_name


def _eval_bool(space: dict, keys: [str]):
    exit_value = None
    for k in keys:
        value = space.get(k, None)
        if value:
            return True
        if value is not None:
            exit_value = False
    return exit_value


def _eval_int(space: dict, keys: [str]):
    exit_value = None
    for k in keys:
        value = space.get(k, None)
        if value is not None:
            if exit_value is None or exit_value < value:
                exit_value = value
    return exit_value


class Policy(object):
    def __init__(self, conf: dict):
        """The authorization policy definition.

        Authorization policy definition with methods to access information about the policy. Init creates the internal
        representation of the policy from a config dictionary.

        Policy evaluation result:

        For bool type of rules or rights:

            True - the rule is satisfied or the right is granted
            False - the rule is not satisfied; the right iis not granted
            None - the rule or right is not applicable (precondition not met)

        For int type or rules or rights:

            Number - the value of the evaluation
            None - the rule or right is not applicable (precondition not met)

        Args:
            conf (dict): the configuration dictionary with keys=groups, users, rights, rules, sites, orgs
        """
        self.config = conf
        self.preconf_valuators = {"selfOrg": self._eval_precond_self_org}

        # compute the rule and right spaces
        self.rule_space = {}
        self.right_space = {}
        groups = conf["groups"]
        for grp_name, grp_def in groups.items():
            rules = grp_def.get("rules", None)
            if rules:
                for rule_name, rule_value in rules.items():
                    key = _group_rule_key(grp_name, rule_name)
                    self.rule_space[key] = rule_value

            role_rights = grp_def.get("role_rights", None)
            if role_rights:
                for role_name, rights in role_rights.items():
                    for right_name, right_value in rights.items():
                        key = _group_role_right_key(grp_name, role_name, right_name)
                        self.right_space[key] = right_value

    def get_config(self):
        return self.config

    def _eval_precond(self, precond: str, user_name: str, org_name: str):
        evaluator = self.preconf_valuators.get(precond, None)
        if not evaluator:
            return None
        else:
            return evaluator(user_name, org_name)

    def _eval_precond_self_org(self, user_name: str, org_name: str):
        users = self.config["users"]
        user = users[user_name]
        return user["org"] == org_name

    def _get_org_groups(self, org_name: str):
        orgs = self.config["orgs"]
        return orgs.get(org_name, None)

    def evaluate_rule_on_org(self, rule_name: str, org_name: str):
        rules = self.config["rules"]
        rule_def = rules.get(rule_name, None)
        if not rule_def:
            return None, 'undefined rule "{}"'.format(rule_name)

        rule_type = rule_def.get("type", "bool")
        groups = self._get_org_groups(org_name)
        if not groups:
            return None, 'unknown org "{}"'.format(org_name)

        keys = []
        for grp_name in groups:
            keys.append(_group_rule_key(grp_name, rule_name))

        if rule_type == "bool":
            result = _eval_bool(space=self.rule_space, keys=keys)
        else:
            result = _eval_bool(space=self.rule_space, keys=keys)

        if result is None:
            result = rule_def["default"]
        return result, ""

    def _get_org_of_site(self, site_name):
        sites = self.config["sites"]
        return sites.get(site_name, None)

    def evaluate_rule_on_site(self, rule_name: str, site_name: str):
        org_name = self._get_org_of_site(site_name)
        if not org_name:
            return None, 'unknown site "{}"'.format(site_name)
        return self.evaluate_rule_on_org(rule_name, org_name)

    def evaluate_user_right_on_org(self, right_name: str, user_name: str, org_name: str):
        rights = self.config["rights"]
        right_def = rights.get(right_name, None)
        if not right_def:
            return None, 'undefined right "{}"'.format(right_name)

        right_type = right_def.get("type", "bool")
        groups = self._get_org_groups(org_name)
        if not groups:
            return None, 'unknown org "{}"'.format(org_name)

        users = self.config["users"]
        user = users.get(user_name, None)
        if not user:
            return None, 'unknown user "{}"'.format(user_name)

        precond = right_def.get("precond", None)
        if precond:
            matched = self._eval_precond(precond, user_name, org_name)
            if not matched:
                if right_type == "bool":
                    return False, ""
                else:
                    return 0, ""

        roles = user["roles"]
        keys = []
        for grp_name in groups:
            for role_name in roles:
                keys.append(_group_role_right_key(grp_name, role_name, right_name))

        if right_type == "bool":
            result = _eval_bool(self.right_space, keys)
        else:
            result = _eval_int(self.right_space, keys)

        if result is None:
            result = right_def["default"]
        return result, ""

    def evaluate_user_right_on_site(self, right_name: str, user_name: str, site_name: str):
        org_name = self._get_org_of_site(site_name)
        if not org_name:
            return None, 'unknown site "{}"'.format(site_name)
        return self.evaluate_user_right_on_org(right_name, user_name, org_name)

    def get_user(self, user_name: str):
        users = self.config["users"]
        return users.get(user_name, None)

    def get_users(self):
        return self.config["users"]

    def get_sites(self):
        return self.config["sites"]

    def get_rights(self):
        return self.config["rights"]

    def get_rules(self):
        return self.config["rules"]

    def get_right_type(self, right_name: str):
        rights = self.config["rights"]
        right_def = rights.get(right_name, None)
        if not right_def:
            return None
        return right_def.get("type", "bool")


class AuthzContext(object):
    def __init__(self, user_name: str, site_names: List[str]):
        """Base class to contain context data for authorization.

        Args:
            user_name (str): user name to be checked
            site_names (List[str]): site names to be checked against
        """
        self.user_name = user_name
        self.site_names = site_names
        self.attrs = {}

    def set_attr(self, key: str, value):
        self.attrs[key] = value

    def get_attr(self, key: str, default=None):
        return self.attrs.get(key, default)


class Authorizer(object):
    def __init__(self):
        """Base class containing the authorization policy."""
        self.policy = None
        self.last_load_time = None

    def get_policy(self) -> Policy:
        return self.policy

    def authorize(self, ctx: AuthzContext) -> (object, str):
        return True, ""

    def evaluate_user_right_on_site(self, right_name: str, user_name: str, site_name: str):
        if not self.policy:
            return None, "policy not defined"
        return self.policy.evaluate_user_right_on_site(right_name=right_name, user_name=user_name, site_name=site_name)

    def evaluate_rule_on_site(self, rule_name: str, site_name: str):
        if not self.policy:
            return None, "policy not defined"

        return self.policy.evaluate_rule_on_site(rule_name=rule_name, site_name=site_name)

    def load_policy(self, policy_config: dict) -> str:
        err = validate_policy_config(policy_config)
        if err:
            return err

        self.policy = Policy(policy_config)
        self.last_load_time = time.time()
        return ""
