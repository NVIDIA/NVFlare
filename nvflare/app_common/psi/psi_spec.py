# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from abc import ABC, abstractmethod
from typing import List, Optional

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.init_final_component import InitFinalComponent
from nvflare.app_common.psi.psi_writer_spec import PSIWriter
from nvflare.app_common.utils.component_utils import check_component_type


class PSI(InitFinalComponent, ABC):
    """The PSI interface is intended for the end-user interface to
    get an intersection of items of clients specified without knowing::

        1) the details of PSI algorithms
        2) real client's own items to other FL clients and FL Servers
    """

    def __init__(self, psi_writer_id: str):
        """
        Args:
            psi_writer_id: a PSIWriter id, we will use it to get PSIWriter from engine.
                PSIWriter will be used save the intersection results.
                For example, FilePSIWriter implements PSIWriter interface and save to the local disk
        """
        super().__init__()
        self.psi_writer_id = psi_writer_id
        self.psi_writer: Optional[PSIWriter] = None
        self.fl_ctx = None
        self.intersection: Optional[List[str]] = None

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        engine = fl_ctx.get_engine()
        psi_writer: PSIWriter = engine.get_component(self.psi_writer_id)
        check_component_type(psi_writer, PSIWriter)
        self.psi_writer = psi_writer

    @abstractmethod
    def load_items(self) -> List[str]:
        """This method needs to be implemented to provide the list of items to PSI algorithm in order to
        calculate intersection.

        Returns: List of Items to be used for intersection calculation

        """
        pass

    def get_intersection(self) -> Optional[List[str]]:
        """This method will return the calculated intersection once PSI job is completed and successful.

        Returns: Intersection result or None
        """
        return self.intersection

    def save(self, intersection: List[str]):
        self.intersection = intersection
        if self.psi_writer:
            self.psi_writer.save(intersection=intersection, overwrite_existing=True, fl_ctx=self.fl_ctx)

    def finalize(self, fl_ctx: FLContext):
        pass
