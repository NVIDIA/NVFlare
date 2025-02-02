class ErrorResponse:
    def __init__(self, status: str, message: str = None, details: dict = None):

        self._status = status
        self._message = message
        self._details = details

    @property
    def status(self) -> str:
        """Gets the status of this ErrorResponse.


        :return: The status of this ErrorResponse.
        :rtype: str
        """
        return self._status

    @status.setter
    def status(self, status: str):
        """Sets the status of this ErrorResponse.


        :param status: The status of this ErrorResponse.
        :type status: str
        """

        self._status = status

    @property
    def message(self) -> str:
        """Gets the message of this ErrorResponse.


        :return: The message of this ErrorResponse.
        :rtype: str
        """
        return self._message

    @message.setter
    def message(self, message: str):
        """Sets the message of this ErrorResponse.


        :param message: The message of this ErrorResponse.
        :type message: str
        """

        self._message = message

    @property
    def details(self) -> dict:
        """Gets the details of this ErrorResponse.


        :return: The details of this ErrorResponse.
        :rtype: object
        """
        return self._details

    @details.setter
    def details(self, details: dict):
        """Sets the details of this ErrorResponse.


        :param details: The details of this ErrorResponse.
        :type details: dict
        """

        self._details = details
