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

from abc import ABC, abstractmethod
from typing import List, Optional

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.psi.psi_persistor import PsiPersistor
from nvflare.app_common.utils.component_utils import check_component_type


class PSI(FLComponent, ABC):
    """
    PSI interface is intended for end-user interface to
    get intersect without knowing the details of PSI algorithms, which will be delegated to the PSIHandler.
    """

    def __init__(self, psi_writer_id: str):
        super().__init__()
        self.psi_writer_id = psi_writer_id
        self.psi_writer: Optional[PsiPersistor] = None
        self.fl_ctx = None

    def initialize(self, fl_ctx: FLContext):
        self.fl_ctx = fl_ctx
        engine = fl_ctx.get_engine()
        psi_writer: PsiPersistor = engine.get_component(self.psi_writer_id)
        check_component_type(psi_writer, PsiPersistor)
        self.psi_writer = psi_writer

    @abstractmethod
    def load_items(self) -> List[str]:
        pass

    def save(self, intersections: List[str]):
        self.psi_writer.save(items=intersections, overwrite_existing=True, fl_ctx=self.fl_ctx)
