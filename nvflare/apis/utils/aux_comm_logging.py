import logging

from nvflare.app_common.widgets.streaming import LogSender


def aux_comm_logging(logger, message):
    """
    LogSender sends the logging data to the server. It must remove itself when calling the logger function
    during the process.
    Args:
        logger: logger object
        message: message to be logged
    Returns:

    """
    log_senders = []
    for handler in logging.root.handlers:
        if isinstance(handler, LogSender):
            log_senders.append(handler)
    for handler in log_senders:
        logger.root.removeHandler(handler)
    logger.info(message)
    for handler in log_senders:
        logger.root.addHandler(handler)
