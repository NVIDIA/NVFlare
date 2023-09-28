from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import NotAuthenticated


class ServerCustomSecurityHandler(FLComponent):
    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.CLIENT_REGISTERED:
            self.authenticate(fl_ctx=fl_ctx)

    def authenticate(self, fl_ctx: FLContext):
        peer_ctx: FLContext = fl_ctx.get_peer_context()
        client_name = peer_ctx.get_identity_name()
        if client_name == "site_b":
            raise NotAuthenticated("site_b not allowed to register")
