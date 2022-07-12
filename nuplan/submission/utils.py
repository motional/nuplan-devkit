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
