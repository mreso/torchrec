#!/bin/bash

set -e

INPUT_PATH=$1
BASE_OUTPUT_PATH=$2

python 01_preproc.py -i $INPUT_PATH -o $BASE_OUTPUT_PATH
python 02_preproc.py -b $BASE_OUTPUT_PATH
python 03_preproc.py -b $BASE_OUTPUT_PATH
