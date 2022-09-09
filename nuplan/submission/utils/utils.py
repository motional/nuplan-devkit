import logging
import socket


def find_free_port_number() -> int:
    """
    Finds a free port number
    :return: the port number of a free port
    """
    skt = socket.socket()
    skt.bind(("", 0))
    port = skt.getsockname()[1]
    skt.close()
    return int(port)


def container_name_from_image_name(image: str) -> str:
    """
    Creates a valid container name from an image name.
    :param image: Docker image name
    :return: A valid container name
    """
    return "_".join(["test", *image.split(":")[0].split("/")])


def get_submission_logger(logger_name: str, logfile: str = '/tmp/submission.log') -> logging.Logger:
    """
    Returns a logger with level WARNING that logs to the given file.
    :param logger_name: Name for the logger.
    :param logfile: Output file for the logger.
    :return: The logger.
    """
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Console output
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # File to be uploaded on EvalAI
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
