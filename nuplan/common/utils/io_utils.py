import asyncio
import logging
import pickle
import tempfile
import uuid
from pathlib import Path
from typing import Any, List, Union, cast

# aiofiles type stubs only suppport python 3.8 or earlier
import aiofiles  # type: ignore
import aiofiles.os  # type: ignore

from nuplan.common.utils.s3_utils import (
    check_s3_object_exists_async,
    check_s3_path_exists_async,
    delete_file_from_s3_async,
    is_s3_path,
    list_files_in_s3_directory_async,
    read_binary_file_contents_from_s3_async,
    split_s3_path,
    upload_file_to_s3_async,
)

logger = logging.getLogger(__name__)


class NuPath(type(Path())):  # type: ignore
    """
    Version of pathlib.Path which handles safe conversions of s3 paths to strings.
    The builtin pathlib.Path converts s3 paths as follows:
        str(Path("s3://a/b/c")) -> "s3:/a/b/c"
    omitting a '/' in the s3 prefix. This can generate errors in downstream functions,
    for example when passing a Path to a pandas io function. This class handles the
    conversion back to string transparently.

    Needs to inherit from type(Path()) because the concrete implementation populates
    a hidden instance variable depending on the platform. For more info, see
    https://stackoverflow.com/a/34116756
    """

    def __str__(self) -> str:
        """
        Override to handle converting s3 paths to strings safely.
        """
        return safe_path_to_string(super().__str__())


async def _save_buffer_async(output_path: Path, buf: bytes) -> None:
    """
    Saves a buffer to file asynchronously.
    The path can either be local or S3.
    :param output_path: The output path to which to save.
    :param buf: The byte buffer to save.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        dump_file = Path(tmp_dir) / f"{str(uuid.uuid4())}.dat" if is_s3_path(output_path) else output_path

        dump_file.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(dump_file, "wb") as f:
            await f.write(buf)

        if is_s3_path(output_path):
            bucket, path = split_s3_path(output_path)
            await upload_file_to_s3_async(dump_file, path, bucket)


def save_buffer(output_path: Path, buf: bytes) -> None:
    """
    Saves a buffer to file synchronously.
    The path can either be local or S3.
    :param output_path: The output path to which to save.
    :param buf: The byte buffer to save.
    """
    asyncio.run(_save_buffer_async(output_path, buf))


def save_object_as_pickle(output_path: Path, obj: Any) -> None:
    """
    Pickles the output object and saves it to the provided path.
    The path can be a local path or an S3 path.
    :param output_path: The output path to which to save.
    :param obj: The object to save. Must be picklable.
    """
    asyncio.run(save_object_as_pickle_async(output_path, obj))


async def save_object_as_pickle_async(output_path: Path, obj: Any) -> None:
    """
    Pickles the output object and saves it to the provided path asynchronously.
    The path can be a local path or an S3 path.
    :param output_path: The output path to which to save.
    :param obj: The object to save. Must be picklable.
    """
    buf = pickle.dumps(obj)
    await _save_buffer_async(output_path, buf)


def save_text(output_path: Path, text: str) -> None:
    """
    Saves the provided text string to the given output path.
    The path can be a local path or an S3 path.
    :param output_path: The output path to which to save.
    :param obj: The text to save.
    """
    asyncio.run(save_text_async(output_path, text))


async def save_text_async(output_path: Path, text: str) -> None:
    """
    Saves the provided text string to the given output path asynchronously.
    The path can be a local path or an S3 path.
    :param output_path: The output path to which to save.
    :param obj: The text to save.
    """
    buf = text.encode("utf-8")
    await _save_buffer_async(output_path, buf)


def read_text(path: Path) -> str:
    """
    Reads a text file from the provided path.
    The path can be a local path or an S3 path.
    :param path: The path to read.
    :return: The text of the file.
    """
    result: str = asyncio.run(read_binary_async(path)).decode("utf-8")
    return result


async def read_text_async(path: Path) -> str:
    """
    Reads a text file from the provided path asynchronously.
    The path can be a local path or an S3 path.
    :param path: The path to read.
    :return: The text of the file.
    """
    binary_content = await read_binary_async(path)
    return binary_content.decode("utf-8")


def read_pickle(path: Path) -> Any:
    """
    Reads an object as a pickle file from the provided path.
    The path can be a local path or an S3 path.
    :param path: The path to read.
    :return: The depickled object.
    """
    return asyncio.run(read_pickle_async(path))


async def read_pickle_async(path: Path) -> Any:
    """
    Reads an object as a pickle file from the provided path asynchronously.
    The path can be a local path or an S3 path.
    :param path: The path to read.
    :return: The depickled object.
    """
    binary_content = await read_binary_async(path)
    return pickle.loads(binary_content)


def read_binary(path: Path) -> bytes:
    """
    Reads binary data from the provided path into memory.
    The path can be a local path or an S3 path.
    :param path: The path to read.
    :return: The contents of the file, in binary format.
    """
    result: bytes = asyncio.run(read_binary_async(path))
    return result


async def read_binary_async(path: Path) -> bytes:
    """
    Reads binary data from the provided path into memory.
    The path can be a local path or an S3 path.
    :param path: The path to read.
    :return: The contents of the file, in binary format.
    """
    if is_s3_path(path):
        bucket, s3_path = split_s3_path(path)
        s3_result: bytes = await read_binary_file_contents_from_s3_async(s3_path, bucket)
        return s3_result
    else:
        async with aiofiles.open(path, "rb") as f:
            local_results: bytes = await f.read()
            return local_results


def path_exists(path: Path, include_directories: bool = True) -> bool:
    """
    Checks to see if a path exists.
    The path can be a local path or an S3 path.
    This method does not examine the file contents.
        That is, a file that exists and empty will return True.
    :param path: The path to check for existance.
    :param include_directories: Whether or not directories count as paths.
    :return: True if the path exists, False otherwise.
    """
    result: bool = asyncio.run(path_exists_async(path, include_directories=include_directories))
    return result


async def path_exists_async(path: Path, include_directories: bool = True) -> bool:
    """
    Checks to see if a path exists.
    The path can be a local path or an S3 path.
    This method does not examine the file contents.
        That is, a file that exists and empty will return True.
    :param path: The path to check for existance.
    :param include_directories: Whether or not directories count as paths.
    :return: True if the path exists, False otherwise.
    """
    if is_s3_path(path):
        bucket, s3_path = split_s3_path(path)
        if include_directories:
            s3_result = await check_s3_path_exists_async(safe_path_to_string(path))
        else:
            s3_result = await check_s3_object_exists_async(s3_path, bucket)
        return cast(bool, s3_result)
    else:
        return path.exists() and (include_directories or path.is_file())


def list_files_in_directory(path: Path) -> List[Path]:
    """
    Returns a list of the string file paths in a directory.
    The path can be a local path or an S3 path.
    :param path: The path to list.
    :return: List of file paths in the folder.
    """
    result: List[Path] = asyncio.run(list_files_in_directory_async(path))
    return result


async def list_files_in_directory_async(path: Path) -> List[Path]:
    """
    Returns a list of the string file paths in a directory.
    The path can be a local path or an S3 path.
    :param path: The path to list.
    :return: List of file paths in the folder.
    """
    if is_s3_path(path):
        bucket, s3_path = split_s3_path(path)
        s3_files = await list_files_in_s3_directory_async(s3_path, bucket)
        with_bucket = [Path(f"s3://{bucket}/{filepath}") for filepath in s3_files]
        return with_bucket
    else:
        return list(path.iterdir())


def delete_file(path: Path) -> None:
    """
    Deletes a single file.
    The path can be a local path or an S3 path.
    :param path: Path of file to delete.
    """
    asyncio.run(delete_file_async(path))


async def delete_file_async(path: Path) -> None:
    """
    Deletes a single file.
    The path can be a local path or an S3 path.
    :param path: Path of file to delete.
    """
    if is_s3_path(path):
        bucket, s3_path = split_s3_path(path)
        await delete_file_from_s3_async(s3_path, bucket)
    else:
        if path.is_dir():
            raise ValueError(f"Expected path {path} to be a file, but got a directory.")

        if await aiofiles.os.path.exists(path):
            await aiofiles.os.unlink(path)


def safe_path_to_string(path: Union[Path, str]) -> str:
    """
    Converts local/s3 paths from Path objects to string.
    It's not always safe to pass the path object to certain io functions.
    For example,
        pd.read_csv(Path("s3://foo/bar"))
    gets interpreted like
        pd.read_csv("s3:/foo/bar")  -- should be s3://, not s3:/
    which is not recognized as an s3 path and raises and error. This function takes a path
    and returns a string that can be passed to any of these functions.
    :param s3_path: Path object of path
    :return: path with the correct format as a string.
    """
    if is_s3_path(path):
        return f's3://{str(path).lstrip("s3:/")}'
    return str(path)
