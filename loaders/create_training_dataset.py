import bisect
import warnings
from operator import itemgetter
from typing import Iterable, List, Optional, TypeVar

import torch
from torch.utils.data import Dataset, DistributedSampler, IterableDataset, Sampler

from .raft import RAFTExhaustiveDataset

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


dataset_dict = {
    'flow': RAFTExhaustiveDataset,
}


class DatasetFromSampler(Dataset):
    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    def __init__(self, sampler, num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True):
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class ConcatDataset(Dataset[T_co]):
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), 'ConcatDataset does not support IterableDataset'
        self.cumulative_sizes = self.cumsum(self.datasets)

    def increase_max_interval_by(self, increment):
        for dataset in self.datasets:
            curr_max_interval = dataset.max_interval.value
            dataset.max_interval.value = min(curr_max_interval + increment, dataset.num_imgs - 1)

    def set_max_interval(self, max_interval):
        for dataset in self.datasets:
            dataset.max_interval.value = min(max_interval, dataset.num_imgs - 1)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError('absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn('cummulative_sizes attribute is renamed to cumulative_sizes', DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


def _split_dataset_types(dataset_types):
    if '+' in dataset_types:
        return dataset_types.split('+')
    if ',' in dataset_types:
        return [item.strip() for item in dataset_types.split(',') if item.strip()]
    return [dataset_types]


def get_training_dataset(args, max_interval):
    dataset_types = _split_dataset_types(args.dataset_types)
    unsupported = [dataset_type for dataset_type in dataset_types if dataset_type != 'flow']
    if unsupported:
        raise ValueError('Unsupported training dataset(s): {}. Use flow for training; keypoints are query-only in this setup.'.format(', '.join(unsupported)))

    if len(dataset_types) == 1:
        train_dataset = dataset_dict['flow'](args, max_interval=max_interval)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    else:
        train_datasets = [dataset_dict['flow'](args, max_interval=max_interval) for _ in dataset_types]
        train_dataset = ConcatDataset(train_datasets)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None

    return train_dataset, train_sampler
