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

from typing import List

from nvflare.fuel.hci.server.authz import Authorizer, AuthzContext


class Right(object):

    UPLOAD_APP = "upload_app"
    DEPLOY_ALL = "deploy_all"
    DEPLOY_SELF = "deploy_self"
    OPERATE_ALL = "operate_all"
    OPERATE_SELF = "operate_self"
    VIEW_ALL = "view_all"
    VIEW_SELF = "view_self"
    TRAIN_ALL = "train_all"
    TRAIN_SELF = "train_self"


class Rule(object):

    ALLOW_BYOC = "allow_byoc"
    ALLOW_CUSTOM_DATALIST = "allow_custom_datalist"


class Action(object):

    UPLOAD = "upload"
    DEPLOY = "deploy"
    OPERATE = "operate"
    VIEW = "view"
    TRAIN = "train"
    BYOC = "byoc"
    CUSTOM_DATALIST = "custom_datalist"


ACTION_EXPLANATION = {
    Action.UPLOAD: 'you are not authorized to upload app to "{site}"',
    Action.DEPLOY: 'you are not authorized to deploy app to "{site}"',
    Action.OPERATE: 'you are not authorized to operate site "{site}"',
    Action.VIEW: 'you are not authorized to view info of site "{site}"',
    Action.TRAIN: 'you are not authorized for training actions on site "{site}"',
    Action.BYOC: 'the app contains custom code, which is not allowed on site "{site}"',
    Action.CUSTOM_DATALIST: 'the app contains custom data list, which is not allowed on site "{site}"',
}


class FLAuthzContext(AuthzContext):
    def __init__(self, user_name: str, site_names: List[str], actions: List[str]):
        """System-wide authorization context.

        Information about the authorization, such as roles, users, sites and actions

        Args:
            user_name (str): user name
            site_names (List[str]): all the sites to be checked
            actions (List[str]): associated actions
        """
        AuthzContext.__init__(self, user_name=user_name, site_names=site_names)
        self.actions = actions

    @staticmethod
    def new_authz_context(site_names: List[str], actions: List[str]):
        assert len(actions) > 0, "actions must be specified"
        for a in actions:
            assert a in [
                Action.UPLOAD,
                Action.DEPLOY,
                Action.OPERATE,
                Action.VIEW,
                Action.TRAIN,
                Action.BYOC,
                Action.CUSTOM_DATALIST,
            ], "invalid action {}".format(a)

        return FLAuthzContext(user_name="", site_names=site_names, actions=actions)


def action_checker_signature(user_name, site_name):
    return True, ""


class FLAuthorizer(Authorizer):
    def __init__(self):
        """System-wide authorization class.

        Examine if a user has certain rights on a specific site
        based on authorization.json file.

        """
        Authorizer.__init__(self)
        self.action_checkers = {
            Action.UPLOAD: self._user_can_upload,
            Action.DEPLOY: self._user_can_deploy,
            Action.OPERATE: self._user_can_operate,
            Action.VIEW: self._user_can_view,
            Action.TRAIN: self._user_can_train,
            Action.BYOC: self._site_allows_byoc,
            Action.CUSTOM_DATALIST: self._site_allows_custom_datalist,
        }

    def evaluate_user_right_on_site(self, right_name: str, user_name: str, site_name: str):
        """Check whether a user has a right in an org.

        Superuser has all rights in all orgs.

        Args:
            right_name: right to be evaluated
            user_name: user to be evaluated against
            site_name: the org

        Returns:
            A tuple of (result, error).

            result: True/False for bool type right; Int number for int rule; None if error occurred during evaluation

            error: Error occurred during evaluation
        """
        right_type = self.policy.get_right_type(right_name)
        if right_type == "bool":
            user = self.policy.get_user(user_name)
            if not user:
                return None, 'unknown user "{}"'.format(user_name)
            roles = user["roles"]
            if "super" in roles:
                # superuser has all rights!
                return True, ""

        return super(FLAuthorizer, self).evaluate_user_right_on_site(right_name, user_name, site_name)

    def _any_bool_rights(self, right_names: List[str], user_name: str, site_name: str):
        if not self.policy:
            return None, "policy not defined"

        user = self.policy.get_user(user_name)
        if not user:
            return None, 'unknown user "{}"'.format(user_name)

        roles = user["roles"]
        if "super" in roles:
            # superuser has all rights!
            return True, ""

        for right_name in right_names:
            value, err = self.evaluate_user_right_on_site(right_name, user_name, site_name)
            if err:
                return None, err

            if value:
                return True, ""
        return False, ""

    def _any_bool_rules(self, rule_names: List[str], site_name: str):
        if not self.policy:
            return None, "policy not defined"

        for rule_name in rule_names:
            value, err = self.evaluate_rule_on_site(rule_name, site_name)
            if err:
                return None, err
            if value:
                return True, ""
        return False, ""

    def _user_can_upload(self, user_name, site_name):
        # must follow action_checker_signature
        return self._any_bool_rights(user_name=user_name, site_name=site_name, right_names=[Right.UPLOAD_APP])

    def _user_can_deploy(self, user_name, site_name):
        # must follow action_checker_signature
        return self._any_bool_rights(
            user_name=user_name, site_name=site_name, right_names=[Right.DEPLOY_ALL, Right.DEPLOY_SELF]
        )

    def _user_can_train(self, user_name, site_name):
        # must follow action_checker_signature
        return self._any_bool_rights(
            user_name=user_name, site_name=site_name, right_names=[Right.TRAIN_ALL, Right.TRAIN_SELF]
        )

    def _user_can_operate(self, user_name, site_name):
        # must follow action_checker_signature
        return self._any_bool_rights(
            user_name=user_name, site_name=site_name, right_names=[Right.OPERATE_ALL, Right.OPERATE_SELF]
        )

    def _user_can_view(self, user_name, site_name):
        # must follow action_checker_signature
        return self._any_bool_rights(
            user_name=user_name, site_name=site_name, right_names=[Right.VIEW_ALL, Right.VIEW_SELF]
        )

    def _site_allows_byoc(self, user_name, site_name):
        # must follow action_checker_signature
        return self._any_bool_rules(site_name=site_name, rule_names=[Rule.ALLOW_BYOC])

    def _site_allows_custom_datalist(self, user_name, site_name):
        # must follow action_checker_signature
        return self._any_bool_rules(site_name=site_name, rule_names=[Rule.ALLOW_CUSTOM_DATALIST])

    def authorize(self, ctx: FLAuthzContext):
        if not ctx:
            return True, ""

        assert isinstance(ctx, FLAuthzContext), "authz_ctx must be FLAuthzContext but got {}".format(type(ctx))

        if not ctx.actions:
            return True, ""

        assert isinstance(ctx.user_name, str) and len(ctx.user_name) > 0, "program error: no user name in ctx!"

        if not ctx.site_names:
            return True, ""

        for action in ctx.actions:
            checker = self.action_checkers.get(action, None)
            if not checker:
                return False, "program error: invalid action name {}".format(action)

            for site_name in ctx.site_names:
                authorized, err = checker(user_name=ctx.user_name, site_name=site_name)
                if err:
                    return False, err

                if not authorized:
                    text_vars = {"site": site_name, "user": ctx.user_name}
                    return False, ACTION_EXPLANATION[action].format(**text_vars)

        return True, ""


class EmptyAuthorizer(Authorizer):
    def authorize(self, ctx: AuthzContext) -> (object, str):
        return True, ""

    def evaluate_user_right_on_site(self, right_name: str, user_name: str, site_name: str):
        return True, ""
