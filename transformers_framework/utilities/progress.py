import os

from tqdm import tqdm


class DownloadProgress(object):

    def __init__(self, total: int = None, name: str = None):
        super().__init__()
        name = "Downloading" if name is None else f"Downloading {os.path.split(name)[-1]}"
        self.progress = tqdm(desc=name, total=total, unit="B", unit_scale=True)

    def __enter__(self):
        return self.callback

    def callback(self, chunk):
        self.progress.update(chunk)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.close()
