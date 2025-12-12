# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from .call_opt import CallOption
from .ctx import get_call_context
from .group import group


class ProxyList(list):

    def __init__(self, proxies: list):
        super().__init__(proxies)

    def __getattr__(self, func_name):
        """This is called to invoke the specified func without specifying call option.
        In this case, default call option will be used.

        Args:
            func_name:

        Returns:

        """

        def method(*args, **kwargs):
            grp = group(
                ctx=get_call_context(),
                proxies=self,
            )
            return getattr(grp, func_name)(*args, **kwargs)

        return method

    def __call__(
        self,
        blocking: bool = True,
        expect_result: bool = True,
        timeout: float = 5.0,
        optional: bool = False,
        secure: bool = False,
        target=None,
        parallel=0,
        process_resp_cb=None,
        **cb_kwargs,
    ):
        """This is called to define the behavior (Call Option) of the group call.

        Args:
            blocking:
            expect_result:
            timeout:
            optional:
            secure:
            target:
            parallel:
            process_resp_cb:
            **cb_kwargs:

        Returns:

        """
        return group(
            ctx=get_call_context(),
            proxies=self,
            call_opt=CallOption(
                blocking=blocking,
                expect_result=expect_result,
                timeout=timeout,
                optional=optional,
                secure=secure,
                target=target,
                parallel=parallel,
            ),
            process_resp_cb=process_resp_cb,
            **cb_kwargs,
        )
