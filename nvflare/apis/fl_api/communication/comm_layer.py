from abc import ABC
from typing import List, Any, Tuple, Optional, Dict, AsyncIterable


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

    def collect_updates(self, site_ids: List[str]) -> List[Any]:
        """
        Synchronously request and collect updates from sites.
        """
        raise NotImplementedError

    async def receive_updates_async(self, site_ids: List[str]) -> AsyncIterable[Tuple[str, Any]]:
        """
        Asynchronously receive site updates as they arrive.
        Yields (site_id, update).
        """
        # for site_id in site_ids:
        #     update = await self._receive_update_from(site_id)
        #     yield site_id, update

        raise NotImplementedError

    def broadcast_state(
            self,
            site_ids: List[str],
            state: Any,
            exclude: Optional[List[str]] = None
    ) -> None:
        """
        Broadcast a state or data to multiple sites, optionally excluding some.
        """
        raise NotImplementedError

    def receive_state(self, site_id: str) -> Any:
        """
        Receive updated state from a specific site.
        Used in cyclic or split workflows.
        """
        raise NotImplementedError

    def receive_all_states(self) -> Any:
        """
        Receive updated state from all sites
        Used in cyclic or split workflows.
        """
        raise NotImplementedError

    def send_message_to_peers(
            self,
            sender_id: str,
            recipient_ids: List[str],
            message_type: str,
            payload: Any,
            timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Send a message from sender to multiple recipients.

        Returns a dict mapping recipient_id to their response or error.
        """
        raise NotImplementedError
