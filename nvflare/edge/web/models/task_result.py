class TaskResult:
    def __init__(
        self,
        session_id: str,
        task_name: str = None,
        result_type: str = None,
        result_data: dict = None,
        device_state: dict = None,
    ):  # noqa: E501
        self._session_id = session_id
        self._task_name = task_name
        self._result_type = result_type
        self._result_data = result_data
        self._device_state = device_state

    @property
    def session_id(self) -> str:
        """Gets the session_id of this TaskResult.


        :return: The session_id of this TaskResult.
        :rtype: str
        """
        return self._session_id

    @session_id.setter
    def session_id(self, session_id: str):
        """Sets the session_id of this TaskResult.


        :param session_id: The session_id of this TaskResult.
        :type session_id: str
        """

        self._session_id = session_id

    @property
    def task_name(self) -> str:
        """Gets the task_name of this TaskResult.


        :return: The task_name of this TaskResult.
        :rtype: str
        """
        return self._task_name

    @task_name.setter
    def task_name(self, task_name: str):
        """Sets the task_name of this TaskResult.


        :param task_name: The task_name of this TaskResult.
        :type task_name: str
        """

        self._task_name = task_name

    @property
    def result_type(self) -> str:
        """Gets the result_type of this TaskResult.


        :return: The result_type of this TaskResult.
        :rtype: str
        """
        return self._result_type

    @result_type.setter
    def result_type(self, result_type: str):
        """Sets the result_type of this TaskResult.


        :param result_type: The result_type of this TaskResult.
        :type result_type: str
        """

        self._result_type = result_type

    @property
    def result_data(self) -> dict:
        """Gets the result_data of this TaskResult.


        :return: The result_data of this TaskResult.
        :rtype: dict
        """
        return self._result_data

    @result_data.setter
    def result_data(self, result_data: dict):
        """Sets the result_data of this TaskResult.


        :param result_data: The result_data of this TaskResult.
        :type result_data: dict
        """

        self._result_data = result_data

    @property
    def device_state(self) -> dict:
        """Gets the device_state of this TaskResult.


        :return: The device_state of this TaskResult.
        :rtype: dict
        """
        return self._device_state

    @device_state.setter
    def device_state(self, device_state: dict):
        """Sets the device_state of this TaskResult.


        :param device_state: The device_state of this TaskResult.
        :type device_state: dict
        """

        self._device_state = device_state
