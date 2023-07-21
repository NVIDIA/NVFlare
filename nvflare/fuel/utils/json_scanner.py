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

import copy
import logging
from abc import ABC, abstractmethod

from nvflare.fuel.common.excepts import ComponentNotAuthorized, ConfigError
from nvflare.security.logging import secure_format_exception, secure_log_traceback


class Node(object):
    def __init__(self, element):
        """A JSON element with additional data.

        Args:
            element: element to create Node object for
        """
        self.parent = None
        self.element = element
        self.level = 0
        self.key = ""
        self.position = 0
        self.paths = []
        self.processor = None
        self.exit_cb = None  # node_exit_cb_signature(node: Node)
        self.props = {}

    def path(self):
        if len(self.paths) <= 0:
            return ""

        return ".".join(self.paths)

    def parent_element(self):
        if self.parent:
            return self.parent.element
        else:
            return None


def _child_node(node: Node, key, pos, element) -> Node:
    child = Node(element)
    child.processor = node.processor
    child.level = node.level + 1
    child.position = pos
    child.parent = node
    child.paths = copy.copy(node.paths)

    child.key = key
    if pos > 0:
        child.key = "#{}".format(pos)

    child.paths.append(child.key)
    return child


class JsonObjectProcessor(ABC):
    """JsonObjectProcessor is used to process JSON elements by the scan_json() function."""

    @abstractmethod
    def process_element(self, node: Node):
        """This method is called by the scan() function for each JSON element scanned.

        Args:
            node: the node representing the JSON element
        """
        pass


class JsonScanner(object):
    def __init__(self, json_data: dict, location=None):
        """Scanner for processing JSON data.

        Args:
            json_data: dictionary containing json data to scan
            location: location to provide in error messages
        """
        if not isinstance(json_data, dict):
            raise ValueError("json_data must be dict")
        self.location = location
        self.data = json_data
        self.logger = logging.getLogger("JsonScanner")

    def _do_scan(self, node: Node):
        try:
            node.processor.process_element(node)
        except ComponentNotAuthorized as e:
            secure_log_traceback(self.logger)

            if self.location:
                raise ComponentNotAuthorized(
                    "Error processing {} in JSON element {}: path: {}, exception: {}".format(
                        self.location, node.element, node.path(), secure_format_exception(e)
                    )
                )
            else:
                raise ComponentNotAuthorized(
                    "Error in JSON element: {}, path: {}, exception: {}".format(
                        node.element, node.path(), secure_format_exception(e)
                    )
                )

        except Exception as e:
            secure_log_traceback(self.logger)

            if self.location:
                raise ConfigError(
                    "Error processing {} in JSON element {}: path: {}, exception: {}".format(
                        self.location, node.element, node.path(), secure_format_exception(e)
                    )
                )
            else:
                raise ConfigError(
                    "Error in JSON element: {}, path: {}, exception: {}".format(
                        node.element, node.path(), secure_format_exception(e)
                    )
                )

        element = node.element

        if isinstance(element, dict):
            # need to make a copy of the element dict in case the processor modifies the dict
            iter_dict = copy.copy(element)
            for k, v in iter_dict.items():
                self._do_scan(_child_node(node, k, 0, v))
        elif isinstance(element, list):
            for i in range(len(element)):
                self._do_scan(_child_node(node, node.key, i + 1, element[i]))

        if node.exit_cb is not None:
            try:
                node.exit_cb(node)
            except Exception as e:
                if self.location:
                    raise ConfigError(
                        "Error post-processing {} in JSON element: {}, exception: {}".format(
                            self.location, node.path(), secure_format_exception(e)
                        )
                    )
                else:
                    raise ConfigError(
                        "Error post-processing JSON element: {}, exception: {}".format(
                            node.path(), secure_format_exception(e)
                        )
                    )

    def scan(self, processor: JsonObjectProcessor):
        if not isinstance(processor, JsonObjectProcessor):
            raise ValueError(f"processor must be JsonObjectProcessor, but got type {type(processor)}")
        node = Node(self.data)
        node.processor = processor
        self._do_scan(node)
