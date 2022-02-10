from typing import (
    Iterator,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
)
import io

import torch
import torch.utils.data.datapipes as dp
from torchdata.datapipes.iter import S3FileLister, S3FileLoader
from torchdata.datapipes.utils import StreamWrapper
from torchrec.datasets.utils import (
    LoadFiles,
    ReadLinesFromCSV)
from torch.utils.data import IterDataPipe
from torchrec.datasets.criteo import _default_row_mapper

s3_prefixes = ['s3://criteo-dataset/day_0']
dp_s3_urls = S3FileLister(s3_prefixes)
dp_s3_files = S3FileLoader(dp_s3_urls) # outputs in (url, BytesIO)
# more datapipes to convert loaded bytes, e.g.


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
        yield from datapipe

#print(dp_s3_files)
#datapipe = StreamWrapper(dp_s3_files).parse_csv_files(delimiter=' ')
#for d in datapipe: # Start loading data
datapipe = S3CriteoIterDataPipe(dp_s3_files)
datapipe = dp.iter.Batcher(datapipe, 100)
datapipe = dp.iter.Collator(datapipe)
batch = next(iter(datapipe))
print(batch.keys())
