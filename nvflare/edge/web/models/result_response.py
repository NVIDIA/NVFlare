class ResultResponse:
    def __init__(
        self,
        status: str,
        message: str = None,
        session_id: str = None,
        retry_wait: int = None,
        task_name: str = None,
        aggregated_result_type: str = None,
        aggregated_result_data: dict = None,
        device_state: dict = None,
    ):

        self._status = status
        self._message = message
        self._session_id = session_id
        self._retry_wait = retry_wait
        self._task_name = task_name
        self._aggregated_result_type = aggregated_result_type
        self._aggregated_result_data = aggregated_result_data
        self._device_state = device_state

    @property
    def status(self) -> str:
        """Gets the status of this ResultResponse.


        :return: The status of this ResultResponse.
        :rtype: str
        """
        return self._status

    @status.setter
    def status(self, status: str):
        """Sets the status of this ResultResponse.


        :param status: The status of this ResultResponse.
        :type status: str
        """
        allowed_values = ["ok", "done", "retry", "invalid"]
        if status not in allowed_values:
            raise ValueError("Invalid value for `status` ({0}), must be one of {1}".format(status, allowed_values))

        self._status = status

    @property
    def message(self) -> str:
        """Gets the message of this ResultResponse.


        :return: The message of this ResultResponse.
        :rtype: str
        """
        return self._message

    @message.setter
    def message(self, message: str):
        """Sets the message of this ResultResponse.


        :param message: The message of this ResultResponse.
        :type message: str
        """

        self._message = message

    @property
    def session_id(self) -> str:
        """Gets the session_id of this ResultResponse.


        :return: The session_id of this ResultResponse.
        :rtype: str
        """
        return self._session_id

    @session_id.setter
    def session_id(self, session_id: str):
        """Sets the session_id of this ResultResponse.


        :param session_id: The session_id of this ResultResponse.
        :type session_id: str
        """

        self._session_id = session_id

    @property
    def retry_wait(self) -> int:
        """Gets the retry_wait of this ResultResponse.


        :return: The retry_wait of this ResultResponse.
        :rtype: int
        """
        return self._retry_wait

    @retry_wait.setter
    def retry_wait(self, retry_wait: int):
        """Sets the retry_wait of this ResultResponse.


        :param retry_wait: The retry_wait of this ResultResponse.
        :type retry_wait: int
        """

        self._retry_wait = retry_wait

    @property
    def task_name(self) -> str:
        """Gets the task_name of this ResultResponse.


        :return: The task_name of this ResultResponse.
        :rtype: str
        """
        return self._task_name

    @task_name.setter
    def task_name(self, task_name: str):
        """Sets the task_name of this ResultResponse.


        :param task_name: The task_name of this ResultResponse.
        :type task_name: str
        """

        self._task_name = task_name

    @property
    def aggregated_result_type(self) -> str:
        """Gets the aggregated_result_type of this ResultResponse.


        :return: The aggregated_result_type of this ResultResponse.
        :rtype: str
        """
        return self._aggregated_result_type

    @aggregated_result_type.setter
    def aggregated_result_type(self, aggregated_result_type: str):
        """Sets the aggregated_result_type of this ResultResponse.


        :param aggregated_result_type: The aggregated_result_type of this ResultResponse.
        :type aggregated_result_type: str
        """

        self._aggregated_result_type = aggregated_result_type

    @property
    def aggregated_result_data(self) -> dict:
        """Gets the aggregated_result_data of this ResultResponse.


        :return: The aggregated_result_data of this ResultResponse.
        :rtype: dict
        """
        return self._aggregated_result_data

    @aggregated_result_data.setter
    def aggregated_result_data(self, aggregated_result_data: dict):
        """Sets the aggregated_result_data of this ResultResponse.


        :param aggregated_result_data: The aggregated_result_data of this ResultResponse.
        :type aggregated_result_data: dict
        """

        self._aggregated_result_data = aggregated_result_data

    @property
    def device_state(self) -> dict:
        """Gets the device_state of this ResultResponse.


        :return: The device_state of this ResultResponse.
        :rtype: dict
        """
        return self._device_state

    @device_state.setter
    def device_state(self, device_state: dict):
        """Sets the device_state of this ResultResponse.


        :param device_state: The device_state of this ResultResponse.
        :type device_state: dict
        """

        self._device_state = device_state
