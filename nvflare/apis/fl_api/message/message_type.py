from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

from nvflare.app_common.abstract.fl_model import FLModel

@dataclass
class MessageEnvelope:
    payload: Any
    type: Optional[str] = None
    sender: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

# Type aliases for messaging
MessageType = Union[FLModel, MessageEnvelope]