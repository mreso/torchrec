#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import List

from torch import distributed as dist
from torch.utils.data import DataLoader
from torchrec.datasets.criteo import (
    INT_FEATURE_COUNT,
    CAT_FEATURE_COUNT,
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    InMemoryBinaryCriteoIterDataPipe,
)
from torchrec.datasets.random import RandomRecDataset

import torch
from typing import (
    Iterator,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
    Tuple,
)
from torchrec.datasets.criteo import _default_row_mapper
from torch.utils.data import IterDataPipe
from torchdata.datapipes.iter import S3FileLister, S3FileLoader
from torchrec.datasets.utils import ReadLinesFromCSV, safe_cast, Batch
import torch.utils.data.datapipes as dp
import io
import numpy as np
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

STAGES = ["train", "val", "test"]
DAYS = 24



class LoadWithTextIOWrapper(IterDataPipe):
    def __init__(self, paths, **open_kw):
        self.paths = paths
        self.open_kw: Any = open_kw  # pyre-ignore[4]

    def __iter__(self) -> Iterator[Any]:
        for url, buffer in self.paths:
            yield url, io.TextIOWrapper(buffer, encoding='utf-8')

class S3CriteoIterDataPipe(IterDataPipe):
    """
    IterDataPipe that can be used to stream either the Criteo 1TB Click Logs Dataset
    (https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) or the
    Kaggle/Criteo Display Advertising Dataset
    (https://www.kaggle.com/c/criteo-display-ad-challenge/) from the source TSV
    files.
    Args:
        paths (Iterable[str]): local paths to TSV files that constitute the Criteo
            dataset.
        row_mapper (Optional[Callable[[List[str]], Any]]): function to apply to each
            split TSV line.
        open_kw: options to pass to underlying invocation of
            iopath.common.file_io.PathManager.open.
    Example:
        >>> datapipe = CriteoIterDataPipe(
        >>>     ("/home/datasets/criteo/day_0.tsv", "/home/datasets/criteo/day_1.tsv")
        >>> )
        >>> datapipe = dp.iter.Batcher(datapipe, 100)
        >>> datapipe = dp.iter.Collator(datapipe)
        >>> batch = next(iter(datapipe))
    """

    def __init__(
        self,
        paths: S3FileLoader,
        *,
        # pyre-ignore[2]
        row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
        # pyre-ignore[2]
        **open_kw,
    ) -> None:
        self.paths = paths
        self.row_mapper = row_mapper
        self.open_kw: Any = open_kw  # pyre-ignore[4]

        batch_size = open_kw['batch_size']

         # These values are the same for the KeyedJaggedTensors in all batches, so they
        # are computed once here. This avoids extra work from the KeyedJaggedTensor sync
        # functions.
        self._num_ids_in_batch: int = CAT_FEATURE_COUNT * batch_size
        self.keys: List[str] = DEFAULT_CAT_NAMES
        self.lengths: torch.Tensor = torch.ones(
            (self._num_ids_in_batch,), dtype=torch.int32
        )
        self.offsets: torch.Tensor = torch.arange(
            0, self._num_ids_in_batch + 1, dtype=torch.int32
        )
        self.stride = batch_size
        self.length_per_key: List[int] = CAT_FEATURE_COUNT * [batch_size]
        self.offset_per_key: List[int] = [
            batch_size * i for i in range(CAT_FEATURE_COUNT + 1)
        ]
        self.index_per_key: Dict[str, int] = {
            key: i for (i, key) in enumerate(self.keys)
        }

    def _np_arrays_to_batch(
        self, dense: np.ndarray, sparse: np.ndarray, labels: np.ndarray
    ) -> Batch:
        return Batch(
            dense_features=torch.from_numpy(dense),
            sparse_features=KeyedJaggedTensor(
                keys=self.keys,
                # transpose + reshape(-1) incurs an additional copy.
                values=torch.from_numpy(sparse.transpose(1, 0).reshape(-1)),
                lengths=self.lengths,
                offsets=self.offsets,
                stride=self.stride,
                length_per_key=self.length_per_key,
                offset_per_key=self.offset_per_key,
                index_per_key=self.index_per_key,
            ),
            labels=torch.from_numpy(labels.reshape(-1)),
        )

    # pyre-ignore[3]
    def __iter__(self) -> Iterator[Any]:
        worker_info = torch.utils.data.get_worker_info()
        paths = self.paths
        if worker_info is not None:
            paths = (
                path
                for (idx, path) in enumerate(paths)
                if idx % worker_info.num_workers == worker_info.id
            )
        # datapipe = LoadFiles(paths, mode="r", **self.open_kw)
        datapipe = LoadWithTextIOWrapper(paths)
        datapipe = ReadLinesFromCSV(datapipe, delimiter="\t")
        if self.row_mapper:
            datapipe = dp.iter.Mapper(datapipe, self.row_mapper)
            # datapipe = dp.iter.Mapper(datapipe, self._np_arrays_to_batch)
        yield from datapipe

class NpToBatchIterDataPipe(IterDataPipe):
    """
    IterDataPipe that can be used to stream either the Criteo 1TB Click Logs Dataset
    (https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) or the
    Kaggle/Criteo Display Advertising Dataset
    (https://www.kaggle.com/c/criteo-display-ad-challenge/) from the source TSV
    files.
    Args:
        paths (Iterable[str]): local paths to TSV files that constitute the Criteo
            dataset.
        row_mapper (Optional[Callable[[List[str]], Any]]): function to apply to each
            split TSV line.
        open_kw: options to pass to underlying invocation of
            iopath.common.file_io.PathManager.open.
    Example:
        >>> datapipe = CriteoIterDataPipe(
        >>>     ("/home/datasets/criteo/day_0.tsv", "/home/datasets/criteo/day_1.tsv")
        >>> )
        >>> datapipe = dp.iter.Batcher(datapipe, 100)
        >>> datapipe = dp.iter.Collator(datapipe)
        >>> batch = next(iter(datapipe))
    """

    def __init__(
        self,
        datapipe,
        batch_size,
        *,
        # pyre-ignore[2]
        row_mapper: Optional[Callable[[List[str]], Any]] = _default_row_mapper,
        # pyre-ignore[2]
        **open_kw,
    ) -> None:
        self.datapipe = datapipe
        self.open_kw: Any = open_kw  # pyre-ignore[4]

         # These values are the same for the KeyedJaggedTensors in all batches, so they
        # are computed once here. This avoids extra work from the KeyedJaggedTensor sync
        # functions.
        self._num_ids_in_batch: int = CAT_FEATURE_COUNT * batch_size
        self.keys: List[str] = DEFAULT_CAT_NAMES
        self.lengths: torch.Tensor = torch.ones(
            (self._num_ids_in_batch,), dtype=torch.int32
        )
        self.offsets: torch.Tensor = torch.arange(
            0, self._num_ids_in_batch + 1, dtype=torch.int32
        )
        self.stride = batch_size
        self.length_per_key: List[int] = CAT_FEATURE_COUNT * [batch_size]
        self.offset_per_key: List[int] = [
            batch_size * i for i in range(CAT_FEATURE_COUNT + 1)
        ]
        self.index_per_key: Dict[str, int] = {
            key: i for (i, key) in enumerate(self.keys)
        }

    def _np_arrays_to_batch(
        self, dense: np.ndarray, sparse: np.ndarray, labels: np.ndarray
    ) -> Batch:
        return Batch(
            dense_features=dense,
            sparse_features=KeyedJaggedTensor(
                keys=self.keys,
                # transpose + reshape(-1) incurs an additional copy.
                # values=sparse.transpose(1, 0).reshape(-1),
                values=sparse.reshape(-1),
                lengths=self.lengths,
                offsets=self.offsets,
                stride=self.stride,
                length_per_key=self.length_per_key,
                offset_per_key=self.offset_per_key,
                index_per_key=self.index_per_key,
            ),
            labels=labels.reshape(-1),
        )

    # pyre-ignore[3]
    def __iter__(self) -> Iterator[Any]:
        for data in self.datapipe:
            if dist.get_rank() == 0:
                print(data)
            yield self._np_arrays_to_batch(*data)

def _get_s3_dataloader(
    args: argparse.Namespace,
    stage: str,
    pin_memory: bool,
) -> DataLoader:
    s3_urls = S3FileLister([args.s3_criteo_prefix])
    if dist.get_rank() == 0:
        print(f"urls: {s3_urls}")
    
    def is_final_day(s: str) -> bool:
        return f"day_{DAYS - 1}" in s

    if stage == "train":
        # Train set gets all data except from the final day.
        s3_urls = list(filter(lambda s: not is_final_day(s), s3_urls))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        # Validation set gets the first half of the final day's samples. Test set get
        # the other half.
        s3_urls = list(filter(is_final_day, s3_urls))
        rank = (
            dist.get_rank()
            if stage == "val"
            else dist.get_rank() + dist.get_world_size()
        )
        world_size = dist.get_world_size() * 2

    s3_urls_buffers = S3FileLoader(s3_urls)

    def row_mapper(row: List[str]) -> Tuple[List[int], List[int], int]:
        label = safe_cast(row[0], int, 0)
        dense = [safe_cast(row[i], int, 0) for i in range(1, 1 + INT_FEATURE_COUNT)]
        sparse = [
            int(safe_cast(row[i], str, "0") or "0", 16)
            for i in range(
                1 + INT_FEATURE_COUNT, 1 + INT_FEATURE_COUNT + CAT_FEATURE_COUNT
            )
        ]
        return dense, sparse, label  # pyre-ignore[7]

    datapipe = S3CriteoIterDataPipe(
            s3_urls_buffers,
            row_mapper=row_mapper,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            hashes=args.num_embeddings_per_feature
            if args.num_embeddings is None
            else ([args.num_embeddings] * CAT_FEATURE_COUNT),
        )
    datapipe = dp.iter.Batcher(datapipe, args.batch_size)

    def my_collate(batch):
        buffer = 3*[None,]
        dtypes = [torch.float32, torch.int64, torch.int64]
        for samples in batch:
            for idx, arr in enumerate(samples):
                if buffer[idx] is None:
                    buffer[idx] = torch.as_tensor(arr, dtype=dtypes[idx])
                else:
                    buffer[idx] = torch.vstack((buffer[idx], torch.as_tensor(arr, dtype=dtypes[idx])))

        dense, sparse, labels = buffer
        dense += 3
        dense = torch.log(dense)

        labels = labels.reshape((-1, 1))

        return dense, sparse, labels

    datapipe = dp.iter.Collator(datapipe, collate_fn=my_collate)

    datapipe = NpToBatchIterDataPipe(datapipe, args.batch_size)

    dataloader = DataLoader(
        datapipe,
        batch_size=None,
        pin_memory=pin_memory,
        collate_fn=lambda x: x,
    )
    return dataloader

def _get_random_dataloader(
    args: argparse.Namespace,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        RandomRecDataset(
            keys=DEFAULT_CAT_NAMES,
            batch_size=args.batch_size,
            hash_size=args.num_embeddings,
            hash_sizes=args.num_embeddings_per_feature,
            manual_seed=args.seed,
            ids_per_feature=1,
            num_dense=len(DEFAULT_INT_NAMES),
        ),
        batch_size=None,
        batch_sampler=None,
        pin_memory=pin_memory,
        num_workers=args.num_workers,
    )


def _get_in_memory_dataloader(
    args: argparse.Namespace,
    stage: str,
    pin_memory: bool,
) -> DataLoader:
    files = os.listdir(args.in_memory_binary_criteo_path)

    def is_final_day(s: str) -> bool:
        return f"day_{DAYS - 1}" in s

    if stage == "train":
        # Train set gets all data except from the final day.
        files = list(filter(lambda s: not is_final_day(s), files))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        # Validation set gets the first half of the final day's samples. Test set get
        # the other half.
        files = list(filter(is_final_day, files))
        rank = (
            dist.get_rank()
            if stage == "val"
            else dist.get_rank() + dist.get_world_size()
        )
        world_size = dist.get_world_size() * 2

    stage_files: List[List[str]] = [
        sorted(
            map(
                lambda x: os.path.join(args.in_memory_binary_criteo_path, x),
                filter(lambda s: kind in s, files),
            )
        )
        for kind in ["dense", "sparse", "labels"]
    ]
    dataloader = DataLoader(
        InMemoryBinaryCriteoIterDataPipe(
            *stage_files,  # pyre-ignore[6]
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            hashes=args.num_embeddings_per_feature
            if args.num_embeddings is None
            else ([args.num_embeddings] * CAT_FEATURE_COUNT),
        ),
        batch_size=None,
        pin_memory=pin_memory,
        collate_fn=lambda x: x,
    )
    return dataloader

def get_dataloader(args: argparse.Namespace, backend: str, stage: str) -> DataLoader:
    """
    Gets desired dataloader from dlrm_main command line options. Currently, this
    function is able to return either a DataLoader wrapped around a RandomRecDataset or
    a Dataloader wrapped around an InMemoryBinaryCriteoIterDataPipe.

    Args:
        args (argparse.Namespace): Command line options supplied to dlrm_main.py's main
            function.
        backend (str): "nccl" or "gloo".
        stage (str): "train", "val", or "test".

    Returns:
        dataloader (DataLoader): PyTorch dataloader for the specified options.

    """
    stage = stage.lower()
    if stage not in STAGES:
        raise ValueError(f"Supplied stage was {stage}. Must be one of {STAGES}.")

    pin_memory = (backend == "nccl") if args.pin_memory is None else args.pin_memory

    if args.in_memory_binary_criteo_path is not None:
        return _get_in_memory_dataloader(args, stage, pin_memory)
    elif args.s3_criteo_prefix is not None:
        return _get_s3_dataloader(args, stage, pin_memory)
    else:
        return _get_random_dataloader(args, pin_memory)
