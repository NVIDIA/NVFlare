# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import threading
from typing import Any, Dict, List

from .fl_constant import ReservedKey

_update_lock = threading.Lock()

MASK_STICKY = 1 << 0
MASK_PRIVATE = 1 << 1

V = "value"
M = "mask"


def is_sticky(mask) -> bool:
    return mask & MASK_STICKY > 0


def is_private(mask) -> bool:
    return mask & MASK_PRIVATE > 0


def make_mask(private, sticky):
    mask = 0
    if private:
        mask += MASK_PRIVATE
    if sticky:
        mask += MASK_STICKY
    return mask


def to_string(mask) -> str:
    if is_private(mask):
        result = "private:"
    else:
        result = "public:"

    if is_sticky(mask):
        return result + "sticky"
    else:
        return result + "non-sticky"


class FLContext(object):
    def __init__(self):
        """Init the FLContext.

        The FLContext is used to passed data between FL Components.
        It can be thought of as a dictionary that stores key/value pairs called props (properties).

        Visibility: private props are only visible to local components,
                    public props are also visible to remote components

        Stickiness: sticky props become available in all future FL Contexts,
                    non-sticky props will only be available in the current FL Context

        """
        self.model = None
        self.props = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_prop_keys(self) -> List[str]:
        return list(self.props.keys())

    def public_key_exists(self, key) -> bool:
        return key in self.props and not is_private(self.props[key][M])

    def get_all_public_props(self) -> Dict[str, Any]:
        result = {}
        with _update_lock:
            for k, v in self.props.items():
                if not is_private(v[M]):
                    _, result[k] = self._get_prop(k)
        return result

    def _get_ctx_manager(self):
        p = self.props.get(ReservedKey.MANAGER, None)
        if p:
            return p[V]
        else:
            return None

    def _get_prop(self, key: str) -> (bool, Any):
        """
        Get the prop with the specified key.
        If the property is sticky, its value will be retrieved from the base (the ctx manager)

        Args:
            key: key of the property

        Returns: tuple: whether the property exists, and the value of the prop if exists.

        """
        # check local first
        p = self.props.get(key)
        if p:
            mask = p[M]
            if not is_sticky(mask):
                return True, p[V]

        # either the prop does not exist locally or it is sticky
        # check with the ctx manager
        ctx_manager = self._get_ctx_manager()
        if ctx_manager:
            assert isinstance(ctx_manager, FLContextManager)
            exists, value, mask = ctx_manager.check_sticker(key)
            if exists:
                self.props[key] = {V: value, M: mask}

        if key in self.props:
            return True, self.props[key][V]
        else:
            return False, None

    def set_prop(self, key: str, value, private=True, sticky=True):
        if not isinstance(key, str):
            raise ValueError("prop key must be str, but got {}".format(type(key)))

        with _update_lock:
            mask = make_mask(private, sticky)

            # see whether a prop with the same key is already defined locally in this ctx
            if key in self.props:
                existing_mask = self.props[key][M]
                if mask != existing_mask:
                    self.logger.warning(
                        f"property '{key}' already exists with attributes "
                        f"{to_string(existing_mask)}, cannot change to {to_string(mask)}"
                    )
                    return False

            # if the prop is sticky, also check with ctx manager to make sure it is consistent with existing mask
            if sticky:
                # check attributes
                ctx_manager = self._get_ctx_manager()
                if ctx_manager:
                    assert isinstance(ctx_manager, FLContextManager)
                    exists, _, existing_mask = ctx_manager.check_sticker(key)
                    if exists and mask != existing_mask:
                        self.logger.warning(
                            f"property '{key}' already exists with attributes "
                            f"{to_string(existing_mask)}, cannot change to {to_string(mask)}"
                        )
                        return False
                    ctx_manager.update_sticker(key, value, mask)

            self.props[key] = {V: value, M: mask}
            return True

    def get_prop(self, key, default=None):
        with _update_lock:
            exists, value = self._get_prop(key)
            if exists:
                return value
            else:
                return default

    def get_prop_detail(self, key):
        with _update_lock:
            if key in self.props:
                prop = self.props.get(key)
                mask = prop[M]
                _, value = self._get_prop(key)
                return {V: value, "private": is_private(mask), "sticky": is_sticky(mask)}
            else:
                return None

    def remove_prop(self, key: str):
        if not isinstance(key, str):
            return

        if key.startswith("__"):
            # do not allow removal of reserved props!
            return

        with _update_lock:
            self.props.pop(key, None)

    def __str__(self):
        raw_list = [f"{k}: {type(v[V])}" for k, v in self.props.items()]
        return " ".join(raw_list)

    # some convenience methods
    def _simple_get(self, key: str, default=None):
        p = self.props.get(key)
        return p[V] if p else default

    def get_engine(self, default=None):
        return self._simple_get(ReservedKey.ENGINE, default)

    def get_job_id(self, default=None):
        return self._simple_get(ReservedKey.RUN_NUM, default)

    def get_identity_name(self, default=""):
        return self._simple_get(ReservedKey.IDENTITY_NAME, default=default)

    def set_job_is_unsafe(self, value: bool = True):
        self.set_prop(ReservedKey.JOB_IS_UNSAFE, value, private=True, sticky=True)

    def is_job_unsafe(self):
        return self.get_prop(ReservedKey.JOB_IS_UNSAFE, False)

    def get_run_abort_signal(self):
        return self._simple_get(key=ReservedKey.RUN_ABORT_SIGNAL, default=None)

    def set_peer_context(self, ctx):
        self.put(key=ReservedKey.PEER_CTX, value=ctx, private=True, sticky=False)

    def get_peer_context(self):
        return self._simple_get(key=ReservedKey.PEER_CTX, default=None)

    def set_public_props(self, metadata: dict):
        # remove all public props
        self.props = {k: v for k, v in self.props.items() if is_private(v[M] or is_sticky(v[M]))}

        for key, value in metadata.items():
            self.set_prop(key, value, private=False, sticky=False)

    def sync_sticky(self):
        # no longer needed since sticky props are always synced
        pass

    def put(self, key: str, value, private, sticky):
        """
        Simply put the prop into the fl context without doing sticky property processing
        Args:
            key:
            value:
            private:
            sticky:

        Returns:

        """
        self.props[key] = {V: value, M: make_mask(private, sticky)}

    # implement Context Manager protocol
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # no longer needed since sticky props are always synced
        pass


class FLContextManager(object):
    """FLContextManager manages the creation and updates of FLContext objects for a run.

    NOTE: The engine may create a new FLContextManager object for each RUN!

    """

    def __init__(
        self, engine=None, identity_name: str = "", job_id: str = "", public_stickers=None, private_stickers=None
    ):
        """Init the FLContextManager.

        Args:
            engine: the engine that created this FLContextManager object
            identity_name (str): identity name
            job_id: the job id
            public_stickers: public sticky properties that are copied into or copied from
            private_stickers: private sticky properties that are copied into or copied from
        """
        self.engine = engine
        self.identity_name = identity_name
        self.job_id = job_id
        self._update_lock = threading.Lock()

        self.public_stickers = {}
        self.private_stickers = {}

        if public_stickers and isinstance(public_stickers, dict):
            self.public_stickers.update(public_stickers)

        if private_stickers and isinstance(private_stickers, dict):
            self.private_stickers.update(private_stickers)

    def new_context(self) -> FLContext:
        """Create a new FLContext object.

        Sticky properties are copied from the stickers into the new context.

        Returns: a FLContext object

        """
        ctx = FLContext()
        ctx.put(key=ReservedKey.MANAGER, value=self, private=True, sticky=False)

        # set permanent props
        ctx.put(key=ReservedKey.ENGINE, value=self.engine, private=True, sticky=False)
        ctx.put(key=ReservedKey.RUN_NUM, value=self.job_id, private=False, sticky=True)

        if self.identity_name:
            ctx.put(key=ReservedKey.IDENTITY_NAME, value=self.identity_name, private=False, sticky=False)

        with self._update_lock:
            for k, v in self.public_stickers.items():
                ctx.put(key=k, value=v, sticky=True, private=False)

            for k, v in self.private_stickers.items():
                ctx.put(key=k, value=v, sticky=True, private=True)
        return ctx

    @staticmethod
    def _get_sticker(stickers, key) -> (bool, Any):
        """
        Get sticker with specified key

        Args:
            stickers:
            key:

        Returns: tuple: whether the sticker exists, value of the sticker if exists

        """
        if key in stickers:
            return True, stickers[key]
        else:
            return False, None

    def check_sticker(self, key: str) -> (bool, Any, int):
        """
        Check whether a sticky prop exists in either the public or private group.

        Args:
            key: the key of the sticker to be checked

        Returns: tuple: whether the sticker exists, its value and mask if it exists

        """
        with self._update_lock:
            exists, value = self._get_sticker(self.private_stickers, key)
            if exists:
                return exists, value, make_mask(True, True)
            exists, value = self._get_sticker(self.public_stickers, key)
            if exists:
                return exists, value, make_mask(False, True)
            return False, None, 0

    def update_sticker(self, key: str, value, mask):
        """
        Update the value of a specified sticker.

        Args:
            key: key of the sticker to be updated
            value: value of the sticker
            mask: mask to determine whether the sticker is public or private

        Returns:

        """
        with self._update_lock:
            if is_private(mask):
                stickers = self.private_stickers
            else:
                stickers = self.public_stickers
            stickers[key] = value
