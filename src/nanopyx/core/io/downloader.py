import tqdm
import os
import requests


def download(url: str, file_path: str):
    """Download a file from a url to a file path.
    :param url: url to download from
    :type url: str
    :param file_path: path to save the file to
    :type file_path: str
    """
    if os.path.exists(file_path):
        raise Warning(f"already exists, no need to download: {file_path}")
        return

    if not os.path.exists(os.path.split(file_path)[0]):
        os.mkdir(os.path.split(file_path)[0])

    with open(file_path, "wb") as f:
        response = requests.get(url, stream=True)
        total = response.headers.get("content-length")

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            with tqdm.tqdm(total=total, unit="iB", unit_scale=True) as pbar:
                for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                    downloaded += len(data)
                    f.write(data)
                    pbar.update(len(data))
