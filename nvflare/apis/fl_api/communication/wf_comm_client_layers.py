from typing import Any, List, Optional, Tuple, Dict
from nvflare.apis.fl_api.interfaces.comm_layer import CommunicationLayer
from nvflare.apis.executor import Executor
from nvflare.apis.impl.wf_comm_client import WFCommClient

# Type aliases for clarity
siteOrSiteList = Any  # Replace with actual type if available
MessageType = Any     # Replace with actual type if available


class ClientCommLayer(CommunicationLayer):
    """
    Communication layer for FL Client using Executor and WFCommClient.
    """
    def __init__(self, executor: Executor, comm_client: WFCommClient):
        self.executor = executor
        self.comm_client = comm_client


    def broadcast_and_wait(self, sites: List[str], message: MessageType ) -> Dict[str, MessageType]:
        raise NotImplementedError

    def push_to_peers(self, sender_id: str, recipients: siteOrSiteList, message_type: str, payload: Any, timeout: Optional[float] = None, meta: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[MessageType]]:
        if hasattr(self.comm_client, 'push_to_peers'):
            return self.comm_client.push_to_peers(sender_id, recipients, message_type, payload, timeout, meta)
        raise NotImplementedError
