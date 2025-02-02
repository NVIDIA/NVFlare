class TaskResponse:

    def __init__(
        self,
        status: str,
        session_id: str,
        study_id: str = None,
        job_id: str = None,
        retry_wait: int = None,
        task_name: str = None,
        task_data: dict = None,
        device_state: dict = None,
    ):
        self._status = status
        self._session_id = session_id
        self._study_id = study_id
        self._job_id = job_id
        self._retry_wait = retry_wait
        self._task_name = task_name
        self._task_data = task_data
        self._device_state = device_state

    @property
    def status(self) -> str:
        """Gets the status of this TaskResponse.


        :return: The status of this TaskResponse.
        :rtype: str
        """
        return self._status

    @status.setter
    def status(self, status: str):
        """Sets the status of this TaskResponse.


        :param status: The status of this TaskResponse.
        :type status: str
        """
        allowed_values = ["ok", "retry", "done"]  # noqa: E501
        if status not in allowed_values:
            raise ValueError("Invalid value for `status` ({0}), must be one of {1}".format(status, allowed_values))

        self._status = status

    @property
    def session_id(self) -> str:
        """Gets the session_id of this TaskResponse.


        :return: The session_id of this TaskResponse.
        :rtype: str
        """
        return self._session_id

    @session_id.setter
    def session_id(self, session_id: str):
        """Sets the session_id of this TaskResponse.


        :param session_id: The session_id of this TaskResponse.
        :type session_id: str
        """

        self._session_id = session_id

    @property
    def study_id(self) -> str:
        """Gets the study_id of this TaskResponse.


        :return: The study_id of this TaskResponse.
        :rtype: str
        """
        return self._study_id

    @study_id.setter
    def study_id(self, study_id: str):
        """Sets the study_id of this TaskResponse.


        :param study_id: The study_id of this TaskResponse.
        :type study_id: str
        """

        self._study_id = study_id

    @property
    def job_id(self) -> str:
        """Gets the job_id of this TaskResponse.


        :return: The job_id of this TaskResponse.
        :rtype: str
        """
        return self._job_id

    @job_id.setter
    def job_id(self, job_id: str):
        """Sets the job_id of this TaskResponse.


        :param job_id: The job_id of this TaskResponse.
        :type job_id: str
        """

        self._job_id = job_id

    @property
    def retry_wait(self) -> int:
        """Gets the retry_wait of this TaskResponse.


        :return: The retry_wait of this TaskResponse.
        :rtype: int
        """
        return self._retry_wait

    @retry_wait.setter
    def retry_wait(self, retry_wait: int):
        """Sets the retry_wait of this TaskResponse.


        :param retry_wait: The retry_wait of this TaskResponse.
        :type retry_wait: int
        """

        self._retry_wait = retry_wait

    @property
    def task_name(self) -> str:
        """Gets the task_name of this TaskResponse.


        :return: The task_name of this TaskResponse.
        :rtype: str
        """
        return self._task_name

    @task_name.setter
    def task_name(self, task_name: str):
        """Sets the task_name of this TaskResponse.


        :param task_name: The task_name of this TaskResponse.
        :type task_name: str
        """

        self._task_name = task_name

    @property
    def task_data(self) -> dict:
        """Gets the task_data of this TaskResponse.


        :return: The task_data of this TaskResponse.
        :rtype: dict
        """
        return self._task_data

    @task_data.setter
    def task_data(self, task_data: dict):
        """Sets the task_data of this TaskResponse.


        :param task_data: The task_data of this TaskResponse.
        :type task_data: dict
        """

        self._task_data = task_data

    @property
    def device_state(self) -> dict:
        """Gets the device_state of this TaskResponse.


        :return: The device_state of this TaskResponse.
        :rtype: dict
        """
        return self._device_state

    @device_state.setter
    def device_state(self, device_state: dict):
        """Sets the device_state of this TaskResponse.


        :param device_state: The device_state of this TaskResponse.
        :type device_state: dict
        """

        self._device_state = device_state
