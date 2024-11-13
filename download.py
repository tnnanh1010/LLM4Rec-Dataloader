import os
import requests
import logging
import math
from tqdm import tqdm
import zipfile

log = logging.getLogger(__name__)

def download(url, filename=None, work_directory="./data"):
    """Download a file if it is not already downloaded.

    Args:
        url: URL of the file to download.
        filename: Optional name for the downloaded file. If not provided, it will use the file’s name extracted from the URL.
        work_directory: The directory to save the downloaded file. Default is "./data".

    Returns:
        str: File path of the file downloaded.
    """
    if filename is None:
        filename = url.split("/")[-1]
    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            log.info(f"Downloading {url}")
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024
            num_iterables = math.ceil(total_size / block_size)
            with open(filepath, "wb") as file:
                for data in tqdm(
                    r.iter_content(block_size),
                    total=num_iterables,
                    unit="KB",
                    unit_scale=True,
                ):
                    file.write(data)
        else:
            log.error(f"Problem downloading {url}")
            r.raise_for_status()
    else:
        log.info(f"File {filepath} already downloaded")

    return filepath



def download_path(path=None):
    """Return a path to download data. If `path=None`, then it yields a temporal path that is eventually deleted,
    otherwise the real path of the input.

    Args:
        path (str): Path to download data.

    Returns:
        str: Real path where the data is stored.

    Examples:
        >>> with download_path() as path:
        >>> ... maybe_download(url="http://example.com/file.zip", work_directory=path)

    """
    if path is None:
        tmp_dir = TemporaryDirectory()
        try:
            yield tmp_dir.name
        finally:
            tmp_dir.cleanup()
    else:
        path = os.path.realpath(path)
        yield path


def unzip_file(zip_src, dst_dir, clean_zip_file=False):
    """Unzip a file

    Args:
        zip_src (str): Zip file.
        dst_dir (str): Destination folder.
        clean_zip_file (bool): Whether or not to clean the zip file.
    """
    fz = zipfile.ZipFile(zip_src, "r")
    for file in fz.namelist():
        fz.extract(file, dst_dir)
    if clean_zip_file:
        os.remove(zip_src)

