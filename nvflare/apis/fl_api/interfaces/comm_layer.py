from abc import ABC
from typing import List, Any, Tuple, Optional, Dict, AsyncIterable, Union
from nvflare.apis.fl_api.message.fl_message import FLMessage

#Todo: focus on sync communication for now. to get the correct interface
#todo: Async will be added later

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

    def collect_from_queue(self, sites: Union[Any, List[Any]] = None) -> List[FLMessage]:
        """
        Synchronously request and collect updates from one or multiple sites.
        If sites is None, collect from all.
        Returns a list of FLMessage objects.
        """
        raise NotImplementedError

    def broadcast_to_queue(
            self,
            sites: Union[Any, List[Any]],
            message: FLMessage,
            exclude: Optional[List[Any]] = None
    ) -> None:
        """
        Broadcast an FLMessage to one or multiple sites, optionally excluding some.
        """
        raise NotImplementedError


    def push_to_peers(
            self,
            sender_id: str,
            recipients: Union[Any, List[Any]],
            message_type: str,
            payload: Any,
            timeout: Optional[float] = None,
    ) -> Dict[Any, Any]:
        """
        Send a message from sender to one or multiple recipients.
        `recipients` can be a single recipient object or a list of recipients (IDs, objects, etc).
        Returns a dict mapping recipient to their response or error.
        """
        # Normalize to list internally if needed
        raise NotImplementedError


    def receive_from_peers(self, recipients: Union[Any, List[Any]] = None) -> Any:
        """
        Receive updated state from one or multiple recipients.
        `recipients` can be a single recipient or a list. If None, receive from all.
        Used in cyclic or split workflows.
        """
        # Normalize to list internally if needed
        raise NotImplementedError
