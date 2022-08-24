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

import logging
import threading

from .fl_constant import ReservedKey

_update_lock = threading.Lock()


class FLContext(object):
    MASK_STICKY = 1 << 0
    MASK_PRIVATE = 1 << 1

    @classmethod
    def _is_sticky(cls, mask) -> bool:
        return mask & cls.MASK_STICKY > 0

    @classmethod
    def _is_private(cls, mask) -> bool:
        return mask & cls.MASK_PRIVATE > 0

    @classmethod
    def _matching_private(cls, mask, private) -> bool:
        return (mask & cls.MASK_PRIVATE > 0) == private

    @classmethod
    def _to_string(cls, mask) -> str:
        if cls._is_private(mask):
            result = "private:"
        else:
            result = "public:"

        if cls._is_sticky(mask):
            return result + "sticky"
        else:
            return result + "non-sticky"

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

    def public_key_exists(self, key):
        return key in self.props and not self._is_private(self.props[key]["mask"])

    def get_all_public_props(self):
        return {k: v["value"] for k, v in self.props.items() if not self._is_private(v["mask"])}

    def set_prop(self, key: str, value, private=True, sticky=True):
        if not isinstance(key, str):
            raise ValueError("prop key must be str, but got {}".format(type(key)))

        with _update_lock:
            mask = 0
            if private:
                mask += FLContext.MASK_PRIVATE
            if sticky:
                mask += FLContext.MASK_STICKY
            if key not in self.props or key in self.props and self.props[key]["mask"] == mask:
                self.props[key] = {"value": value, "mask": mask}
            else:
                existing_mask = self.props[key]["mask"]
                self.logger.warning(
                    f"property {key} already exists with attributes "
                    f"{self._to_string(existing_mask)}, cannot change to {self._to_string(mask)}"
                )
                return False
        return True

    def get_prop(self, key, default=None):
        with _update_lock:
            if key in self.props:
                return self.props.get(key)["value"]
            else:
                return default

    def get_prop_detail(self, key):
        with _update_lock:
            if key in self.props:
                prop = self.props.get(key)
                mask = prop["mask"]
                return {"value": prop["value"], "private": self._is_private(mask), "sticky": self._is_sticky(mask)}
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
        raw_list = [f'{k}: {type(v["value"])}' for k, v in self.props.items()]
        return " ".join(raw_list)

    # some convenience methods
    def get_engine(self):
        return self.get_prop(key=ReservedKey.ENGINE, default=None)

    def get_job_id(self):
        return self.get_prop(key=ReservedKey.RUN_NUM, default=None)

    def get_identity_name(self):
        return self.get_prop(key=ReservedKey.IDENTITY_NAME, default="")

    def get_run_abort_signal(self):
        return self.get_prop(key=ReservedKey.RUN_ABORT_SIGNAL, default=None)

    def set_peer_context(self, ctx):
        self.set_prop(key=ReservedKey.PEER_CTX, value=ctx, private=True, sticky=False)

    def get_peer_context(self):
        return self.get_prop(key=ReservedKey.PEER_CTX, default=None)

    def set_public_props(self, metadata: dict):
        # remove all public props
        self.props = {k: v for k, v in self.props.items() if self._is_private(v["mask"] or self._is_sticky(v["mask"]))}

        for key, value in metadata.items():
            self.set_prop(key, value, private=False, sticky=True)

    def clone_sticky(self):
        new_fl_ctx = FLContext()
        for k, v in self.props.items():
            if self._is_sticky(v["mask"]):
                new_fl_ctx.props[k] = {"value": v["value"], "mask": v["mask"]}
        return new_fl_ctx

    def sync_sticky(self):
        ctx_manager = self.get_prop(key=ReservedKey.MANAGER, default=None)
        if not ctx_manager:
            raise ValueError("FLContextManager does not exist.")

        ctx_manager.finalize_context(self)

    # implement Context Manager protocol
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mgr = self.get_prop(key=ReservedKey.MANAGER)
        # if not mgr:
        #     raise ValueError("This FLContext is not managed. Please use FLContextManager to create FLContext")
        if mgr:
            mgr.finalize_context(self)


class FLContextManager(object):
    """FLContextManager manages the creation and updates of FLContext objects for a run.

    NOTE: The engine may create a new FLContextManager object for each RUN!

    """

    def __init__(self, engine, identity_name: str, job_id: str, public_stickers, private_stickers):
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
        ctx.set_prop(key=ReservedKey.MANAGER, value=self, private=True, sticky=False)

        # set permanent props
        ctx.set_prop(key=ReservedKey.ENGINE, value=self.engine, private=True)
        ctx.set_prop(key=ReservedKey.RUN_NUM, value=self.job_id, private=False)

        if self.identity_name:
            ctx.set_prop(key=ReservedKey.IDENTITY_NAME, value=self.identity_name, private=False)

        with self._update_lock:
            for k, v in self.public_stickers.items():
                ctx.set_prop(key=k, value=v, sticky=True, private=False)

            for k, v in self.private_stickers.items():
                ctx.set_prop(key=k, value=v, sticky=True, private=True)

        return ctx

    def finalize_context(self, ctx: FLContext):
        """Finalize the context by copying/updating sticky props into stickers.

        Args:
            ctx: the context to be finalized

        """
        with self._update_lock:
            for k, v in ctx.props.items():
                if ctx._is_sticky(v["mask"]):
                    if ctx._is_private(v["mask"]):
                        self.private_stickers[k] = v["value"]
                    else:
                        self.public_stickers[k] = v["value"]
