from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor


class HubPersistor(LearnablePersistor):
    def load(self, fl_ctx: FLContext) -> Learnable:
        """Load the Learnable object.

        Args:
            fl_ctx: FLContext

        Returns:
            Learnable object loaded

        """
        return Learnable()

    def save(self, learnable: Learnable, fl_ctx: FLContext):
        return
