def not_implemented(message):
    """
    This method raises a ValueError exception with a message explaining what is not implemented

    We encourage the use of this function anticipate all unsupported types or features and give clear messages to the user.

    Args:
        message (str): A helpful message explaining what is not implemented (a given type for an argument, etc.)

    Raises:
        ValueError: exception with a message explaining what is not implemented.

    """

    raise ValueError("mercury.graph does not yet support:" + message)
