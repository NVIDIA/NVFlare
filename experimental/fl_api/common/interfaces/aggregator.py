from typing import List

from experimental.fl_api.common.interfaces.message_type import MessageType


class Aggregator:
    def aggregate(self, updates: List[MessageType]) -> MessageType:
        raise NotImplementedError

