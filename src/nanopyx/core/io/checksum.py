import hashlib


def get_checksum(file_path: str) -> str:
    """
    Returns the SHA-256 checksum of the file
    :param file_path: path to the file
    :type file_path: str
    :return: checksum in hexadecimal format
    :rtype: str
    """

    # Open the file in binary mode
    with open(file_path, "rb") as file:
        # Create a SHA-256 hash object
        hash_object = hashlib.sha256()

        # Iterate over the file in chunks
        for chunk in iter(lambda: file.read(4096), b""):
            # Feed the chunk to the hash object
            hash_object.update(chunk)

    # Obtain the checksum in hexadecimal format
    checksum = hash_object.hexdigest()

    return checksum
