import os

def ensure_dir(directory):
    """
    Ensure a directory exists; if not, create it.

    Args:
        directory (str): Path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")


def read_file(file_path):
    """
    Read the contents of a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Contents of the file.
    """
    with open(file_path, "r") as file:
        return file.read()


def write_file(file_path, content):
    """
    Write content to a file.

    Args:
        file_path (str): Path to the file.
        content (str): Content to write.
    """
    with open(file_path, "w") as file:
        file.write(content)
        print(f"Content written to {file_path}")
