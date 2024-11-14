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



