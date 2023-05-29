from typing import Dict, Sequence

from torch.utils.data import Dataset


class MapDataset(Dataset):
    r""" Superclass of all map datasets. Tokenization is performed on the fly.
    Dataset may be completely loaded into memory. """

    def __init__(self, data: Sequence[Dict]):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        r""" Get dict of data at a given position. """
        res = self.data[idx]
        if 'index' not in res:
            res['index'] = idx
        return res
