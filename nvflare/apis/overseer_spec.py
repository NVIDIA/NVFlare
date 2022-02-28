from abc import ABC


from dataclasses import dataclass

from typing import Any, Dict


@dataclass
class SP:
    name: str = ""
    fl_port: str = ""
    adm_port: str = ""
    service_session_id: str = ""
    primary: bool = False


class OverseerAgent(ABC):
    def initialize(
        self,
        overseer_end_point: str,
        project: str,
        role: str,
        name: str,
        fl_port: str = "",
        adm_port: str = "",
        aux: dict = {},
        *args,
        **kwargs,
    ):
        pass
    def start(self, update_callback=None):
        pass

    def pause(self):
        pass

    def resume(self):
        pass

    def end(self):
        pass

    def set_secure_context(self, ca_path: str, cert_path: str = "", prv_key_path: str = ""):
        pass

    def get_primary_sp(self) -> SP:
        """Return current primary service provider.

        If primary sp not available, such as not reported by SD, connection to SD not established yet
        the name and ports will be empty strings.
        """
        pass

    def add_payload(self, payload: Dict[str, Any]):
        pass

    def get_overseer_status(self) -> Dict[str, Any]:
        """

        Returns:
            Dict[str, Any]: [description]
        """
        pass
