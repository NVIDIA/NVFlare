from abc import ABC
from typing import List, Any, Tuple, Optional, Dict, Union

from nvflare.apis.fl_api.message.fl_message import MessageType

siteOrSiteList = Union[str, List[str]]

class CommunicationLayer(ABC):
    """
    Abstract interface for communication between server and clients
    in federated learning or similar distributed workflows.
    """

    def get_initial_state(self) -> Any:
        """
        Retrieve the initial global state to start training or processing.
        """
        raise NotImplementedError

    def collect_from_queue(self, sites: siteOrSiteList = None) -> List[MessageType]:
        """
        Synchronously request and collect updates from one or multiple sites.
        If sites is None, collect from all.
        """
        raise NotImplementedError

    def send_to_queue(self, site: siteOrSiteList, message: MessageType) -> None:
        """
        Send a message to a specific site or list of sites.
        """
        raise NotImplementedError

    def broadcast_to_queue(
            self,
            sites: List[str],
            message: MessageType,
            exclude: Optional[List[str]] = None
    ) -> None:
        """
        Broadcast a message to multiple sites, with optional exclusions.
        """
        raise NotImplementedError

    def push_to_peers(
            self,
            sender_id: str,
            recipients: siteOrSiteList,
            message_type: str,
            payload: Any,
            timeout: Optional[float] = None,
            meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[MessageType]]:
        """
        Push a payload to recipients with a given message type.
        Returns a tuple of (recipient list, response list).
        """
        raise NotImplementedError

    def receive_from_peers(self, recipients: siteOrSiteList = None) -> Any:
        """
        Receive updated state from one or multiple recipients.
        `recipients` can be a single recipient or a list. If None, receive from all.
        """
        raise NotImplementedError
