"""MIMIC-IV-MM split-level feature loading."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class NPY_Large_MultimodalPred_Dataset(Dataset):
    """Read large/small-model NPY arrays using the original dataset layout.

    All feature arrays share sample order. Large- and small-model labels must
    match elementwise. Feature tensors are float32 and labels are int64.
    """

    def __init__(self, split, task, image_model_name, text_model_name,
                 timeseries_model_name, large_data_path='./data/large_embed',
                 small_data_path='./data/MultimodalPred_input', *, feature_dims=None):
        super().__init__()
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val, or test')
        self.split = split
        self.task = task
        self.large_data_path = Path(large_data_path).expanduser()
        self.small_data_path = Path(small_data_path).expanduser()
        model_name = '_'.join((image_model_name, text_model_name, timeseries_model_name))
        large_task = self.large_data_path / task
        large_features = large_task / model_name
        small_task = self.small_data_path / task

        # Match the six files read by the original MIMIC-IV-MM dataset.
        self.large_img = self._load(large_features / f'{split}_img.npy', torch.float32)
        self.large_text = self._load(large_features / f'{split}_text.npy', torch.float32)
        self.large_time = self._load(large_features / f'{split}_time.npy', torch.float32)
        self.large_label = self._load(large_task / f'{split}_label.npy', torch.long)
        self.small_feat = self._load(small_task / f'{split}_feat.npy', torch.float32)
        self.small_label = self._load(small_task / f'{split}_target.npy', torch.long)

        if self.large_label.ndim != 1 or self.small_label.ndim != 1:
            raise ValueError('Label arrays must have shape [N]')
        if not torch.equal(self.large_label, self.small_label):
            raise ValueError(f'Large- and small-model labels differ in the {split} split')
        self.label = self.large_label
        if not len(self.label):
            raise ValueError(f'The {split} split is empty')
        if not torch.all((self.label == 0) | (self.label == 1)):
            raise ValueError('MIMIC-IV-MM labels must be binary 0/1')

        arrays = {'image': self.large_img, 'text': self.large_text,
                  'time': self.large_time, 'small': self.small_feat}
        for name, tensor in arrays.items():
            if tensor.ndim != 2 or len(tensor) != len(self.label):
                raise ValueError(f'{name} features must have shape [N, D] with N={len(self.label)}')
            expected = (feature_dims or {}).get(name)
            if expected is not None and tensor.shape[1] != expected:
                raise ValueError(f'{name}: expected dimension {expected}, got {tensor.shape[1]}')

    @staticmethod
    def _load(path, dtype):
        array = np.load(path, allow_pickle=False)
        if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
            raise ValueError(f'Expected a finite numeric array: {path}')
        if dtype == torch.long and not np.all((array == 0) | (array == 1)):
            raise ValueError(f'Expected binary 0/1 labels: {path}')
        return torch.tensor(array, dtype=dtype)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return (self.large_img[index], self.large_text[index],
                self.large_time[index], self.small_feat[index], self.label[index])
