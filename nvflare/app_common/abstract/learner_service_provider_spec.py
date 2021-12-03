from abc import ABC, abstractmethod


class LearnerServiceProviderSpec(ABC):
    @abstractmethod
    def initialize(self, parts: dict, ctx: dict):
        """
        Initializes the LearnerServiceProvider object.

        This is called before the LearnerServiceProvider can train or validate.
        This is called only once.

        Args:
            parts (dict): components to be used by the LearnerServiceProvider.
            ctx (dict): contextual info of running environment. For example:
                        MMAR root, security settings, workspace location, stuff in FLContext

        NOTE:
            Some example parts:
                - A "SummaryWriter" part could be implemented as a NVFlare widget that knows how to
                  stream data (fire an event back to NVFlare).
                - A "Logger" part that streams log messages back.
        """
        pass

    @abstractmethod
    def train(self, data: dict, ctx: dict) -> dict:
        """
        Called to perform training.

        This method can be called many times during the life time of the LearnerServiceProvider.

        Args:
            data (dict): the training input data (e.g. model weights)
            ctx (dict): contextual info of running environment.

        Returns:
            A dictionary that contains the training result.
        """
        pass

    @abstractmethod
    def validate(self, data: dict, ctx: dict) -> dict:
        """
        Called to perform validation.

        This method can be called many times during the life time of the LearnerServiceProvider.

        Args:
            data (dict): the validation input data (e.g. model weights)
            ctx (dict): contextual info of running environment.

        Returns:
            A dictionary that contains the validation result.
        """
        pass

    @abstractmethod
    def abort(self, ctx: dict):
        """
        Called (from another thread) to abort the current task (validate or train).

        Args:
            ctx (dict): contextual info of running environment.

        Note:
            This is to abort the current task only, not the LearnerServiceProvider.
            After aborting, the LearnerServiceProvider may still be called to perform another task.
        """
        pass

    @abstractmethod
    def finalize(self, ctx: dict):
        """
        Called to finalize the LearnerServiceProvider (close/release resources gracefully).
        After this call, the LearnerServiceProvider will be destroyed.

        Args:
            ctx (dict): contextual info of running environment.
        """
        pass
