from typing import List, Any, Tuple, Optional, Dict, AsyncIterable


class CommunicationLayer:
    """
    Abstract interface for communication between server and clients
    in federated learning or similar distributed workflows.
    """

    def get_initial_state(self) -> Any:
        """
        Retrieve the initial global state to start training or processing.
        """
        raise NotImplementedError

    def collect_updates(self, client_ids: List[str]) -> List[Any]:
        """
        Synchronously request and collect updates from clients.
        """
        raise NotImplementedError

    async def receive_updates_async(self, client_ids: List[str]) -> AsyncIterable[Tuple[str, Any]]:
        """
        Asynchronously receive client updates as they arrive.
        Yields (client_id, update).
        """
        # for client_id in client_ids:
        #     update = await self._receive_update_from(client_id)
        #     yield client_id, update

        raise NotImplementedError

    def broadcast_state(
            self,
            client_ids: List[str],
            state: Any,
            exclude: Optional[List[str]] = None
    ) -> None:
        """
        Broadcast a state or data to multiple clients, optionally excluding some.
        """
        raise NotImplementedError

    def receive_state(self, client_id: str) -> Any:
        """
        Receive updated state from a specific client.
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

    def disconnect_clients(self, client_ids: List[str]) -> None:
        """
        Optionally disconnect or deregister clients after processing.
        """
        raise NotImplementedError


