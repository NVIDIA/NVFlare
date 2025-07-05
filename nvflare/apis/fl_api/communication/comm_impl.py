from typing import List, Any
from nvflare.apis.fl_api.interfaces.comm_layer import CommunicationLayer
from nvflare.apis.fl_api.message.fl_message import MessageType, MessageEnvelope
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import CellChannel
from nvflare.fuel.utils import fobs

class InMemoryCommunicationLayer(CommunicationLayer):
    """
    Simple in-memory implementation of the CommunicationLayer for simulation/testing.
    Not for production use.
    """
    def __init__(self):
        self.queues = {}

    def broadcast_to_queue(self, recipients: List[str], message: MessageType):
        for r in recipients:
            self.queues.setdefault(r, []).append(message)

    def collect_from_queue(self, recipient: str) -> MessageType:
        queue = self.queues.get(recipient, [])
        if not queue:
            raise RuntimeError(f"No messages for recipient {recipient}")
        return queue.pop(0)

    def push_to_peers(self, sender_id: str, recipients: List[str], message_type: str, payload: Any):
        envelope = MessageEnvelope()
        envelope.payload = {"type": message_type, "payload": payload, "sender": sender_id}
        self.broadcast_to_queue(recipients, envelope)

    def receive_from_peers(self, peer_ids: List[str]) -> List[MessageType]:
        results = []
        for pid in peer_ids:
            queue = self.queues.get(pid, [])
            if queue:
                results.append(queue.pop(0))
        return results

class CellnetCommunicationLayer(CommunicationLayer):
    def __init__(self, cell: Cell, channel: str = CellChannel.SERVER_MAIN, topic: str = "fed_comm"):
        self.cell = cell
        self.channel = channel
        self.topic = topic

    def broadcast_to_queue(self, recipients: List[str], message: MessageType) -> None:
        payload = fobs.dumps(message)
        for peer in recipients:
            self.cell.core_cell.fire_and_forget(
                channel=self.channel,
                topic=self.topic,
                targets=peer,
                message=payload
            )

    def collect_from_queue(self, recipient: str) -> MessageType:
        msg = self.cell.recv_one(recipient, channel=self.channel, topic=self.topic)
        return fobs.loads(msg)

    def push_to_peers(self, sender_id: str, recipients: List[str], message_type: str, payload: Any) -> None:
        envelope = MessageEnvelope()
        envelope.payload = {"type": message_type, "payload": payload, "sender": sender_id}
        self.broadcast_to_queue(recipients, envelope)

    def receive_from_peers(self, peer_ids: List[str]) -> List[MessageType]:
        results = []
        for peer in peer_ids:
            msg = self.cell.recv_one(peer, channel=self.channel, topic=self.topic)
            results.append(fobs.loads(msg))
        return results
