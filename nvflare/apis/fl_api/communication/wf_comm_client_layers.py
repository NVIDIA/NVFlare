from typing import Any, List, Optional, Tuple, Dict
from nvflare.apis.fl_api.interfaces.comm_layer import CommunicationLayer
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.wf_comm_client import WFCommClient

# Type aliases for clarity
from nvflare.apis.wf_comm_spec import WFCommSpec

siteOrSiteList = Any  # Replace with actual type if available
MessageType = Any     # Replace with actual type if available


class ClientCommLayer(CommunicationLayer):
    """
    Communication layer for FL Client using Executor and WFCommClient.
    """
    def __init__(self, communicator: WFCommSpec, fl_ctx: FLContext):
        if communicator is None:
            raise ValueError("communicator must not be None.")
        if fl_ctx is None:
            raise ValueError("fl_ctx must not be None.")

        self.communicator: WFCommSpec = communicator
        self.fl_ctx: FLContext = fl_ctx
        self.response = {}
        self.errors = {}

    @property
    def comm(self) -> WFCommSpec:
        return self.communicator

    def broadcast_and_wait(self, sites: List[str], message: MessageType ) -> Dict[str, MessageType]:
        raise NotImplementedError

    def push_to_peers(self, sender_id: str, recipients: siteOrSiteList, message_type: str, payload: Any, timeout: Optional[float] = None, meta: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[MessageType]]:
         raise NotImplementedError
