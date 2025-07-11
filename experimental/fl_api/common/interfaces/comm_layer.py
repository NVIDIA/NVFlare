from abc import ABC
from typing import List, Tuple, Dict, Union

from experimental.fl_api.common.interfaces.message_type import MessageType

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
