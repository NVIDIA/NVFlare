from abc import ABC
from typing import List, Any, Tuple, Optional, Dict, Union

from nvflare.apis.fl_api.message.message_type import MessageType

siteOrSiteList = Union[str, List[str]]


class CommunicationLayer(ABC):
    """
    Abstract interface for communication between server and clients
    in federated learning or similar distributed workflows.
    """

    def broadcast_and_wait(self, sites: siteOrSiteList, message: MessageType) -> Dict[str, MessageType]:
        """
        Broadcast a message to multiple sites, with optional exclusions.
        Returns the response message.
        """
        raise NotImplementedError

    def push_to_peers(self, recipients: siteOrSiteList, message: MessageType) -> Tuple[List[str], List[MessageType]]:
        """
        Push a payload to recipients with a given message type.
        Returns a tuple of (recipient list, response list).
        """
        raise NotImplementedError


class SimulatedCommLayer(CommunicationLayer):
    def broadcast_and_wait(self, sites: siteOrSiteList, message: MessageType) -> Dict[str, MessageType]:
        if isinstance(sites, str):
            sites = [sites]
        print("[SimulatedCommLayer] Simulating broadcast_and_waitL: returning same message for all sites")
        # Return the same message for each site to match the correct type
        return {site: message for site in sites}
