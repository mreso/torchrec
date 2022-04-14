import argparse
from contextlib import ExitStack
import os
import shutil
import time
from typing import List, TypeVar, Callable, Iterable

import nvtabular as nvt
import numpy as np
import numpy.lib.format as fmt


FREQUENCY_THRESHOLD = 3
INT_FEATURE_COUNT = 13
CAT_FEATURE_COUNT = 26
DAYS = 24
DEFAULT_LABEL_NAME = "label"
DEFAULT_INT_NAMES: List[str] = [f"int_{idx}" for idx in range(INT_FEATURE_COUNT)]
DEFAULT_CAT_NAMES: List[str] = [f"cat_{idx}" for idx in range(CAT_FEATURE_COUNT)]

DEFAULT_COLUMN_NAMES: List[str] = [
    DEFAULT_LABEL_NAME,
    *DEFAULT_INT_NAMES,
    *DEFAULT_CAT_NAMES,
]

dtypes = {c: np.int32 for c in DEFAULT_COLUMN_NAMES[:14] + [DEFAULT_LABEL_NAME]}
dtypes.update({c: 'hex' for c in DEFAULT_COLUMN_NAMES[14:]})

entries_per_day = [
    195841983, 199563535, 196792019, 181115208, 152115810,
    172548507, 204846845, 200801003, 193772492, 198424372,
    185778055, 153588700, 169003364, 194216520, 194081279,
    187154596, 177984934, 163382602, 142061091, 156534237,
    193627464, 192215183, 189747893, 178274637
    ]


class NumpyWriter(object):
    def __init__(self, path: str, dtype: np.dtype, shape: Iterable[int]):
        assert len(shape) == 2, 'Only works with two dimensional arrays'
        self.path = path
        self.dtype = dtype
        self.shape = shape
        self.f = None

    def __enter__(self):
        self.f = open(self.path, 'wb')
        header = {
            'descr': fmt.dtype_to_descr(self.dtype),
            'fortran_order': False,
            'shape': self.shape
            }
        fmt.write_array_header_2_0(self.f, header)
        return self

    def append(self, x: np.array):
        self.f.write(x.tobytes('C'))

    def __exit__(self, *args):
        self.f.close()


def process_criteo_day(input_path, output_path, day):
    config = {
        'engine':'csv',
        'names':DEFAULT_COLUMN_NAMES,
        'part_memory_fraction':1,
        'sep':'\t',
        'dtypes':dtypes,
    }
    
    tsv_dataset = nvt.Dataset(os.path.join(input_path, f'day_{day}'), **config)

    cat_features = DEFAULT_CAT_NAMES >> nvt.ops.FillMissing()
    cont_features = DEFAULT_INT_NAMES >> nvt.ops.FillMissing() >> nvt.ops.LambdaOp(lambda col: col + 2) >> nvt.ops.LogOp()
    features = cat_features + cont_features + [DEFAULT_LABEL_NAME]
    workflow = nvt.Workflow(features)

    workflow.fit(tsv_dataset)

    with ExitStack() as stack:
        dense_writer = stack.enter_context(NumpyWriter(os.path.join(output_path, f'day_{day}_dense.npy'), np.dtype(np.float32), (entries_per_day[day], INT_FEATURE_COUNT)))
        sparse_writer = stack.enter_context(NumpyWriter(os.path.join(output_path, f'day_{day}_sparse.npy'), np.dtype(np.int32), (entries_per_day[day], CAT_FEATURE_COUNT)))
        label_writer = stack.enter_context(NumpyWriter(os.path.join(output_path, f'day_{day}_labels.npy'), np.dtype(np.int32), (entries_per_day[day], 1)))

        for t in workflow.transform(tsv_dataset).to_iter():
            dense_writer.append(t[DEFAULT_INT_NAMES].to_numpy())
            sparse_writer.append(t[DEFAULT_CAT_NAMES].to_numpy())
            label_writer.append(t[DEFAULT_LABEL_NAME].to_numpy())


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess criteo dataset')
    parser.add_argument('--input_path', '-i', dest='input_path', help='Input path containing tsv files')
    parser.add_argument('--output_path', '-o', dest='output_path', help='Output path')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    assert os.path.exists(args.input_path), f'Input path {args.input_path} does not exist'

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for day in range(24):
        start_time = time.time()
        process_criteo_day(args.input_path, args.output_path, day)
        print(f'Day {day} processed in {time.time()-start_time} sec')



