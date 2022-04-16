
# Criteo Preprocessing

The Criteo 1TB Clock Logs dataset is a dataset comprised of 24 days of feature values and click results. More info can be found here: https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/.

The data is stored in tab separated value files which we need to preprocess to fill missing values and apply transformation to compress the value range before we dump them it into a numpy files for storage.

To preprocess the data we are using NVTabular which speeds up the precessing by utilizing the parallel compute powers of our GPUs.

This script assumes that the files of the dataset have already been downloaded and extracted.

It then processes the files in three steps:

1. Convert tsv files into parquet format

2. Apply tranformation

3. Dump values into numpy files

Note that the total amount of disk space needed for this process is about 2.5 TB (without the zipped original files).

For convenience this example used a docker container.

To start the docker run:
 
    cd torchrec/examples/datasets/criteo_preprocessing

    docker build -t criteo_preprocessing:latest . && \
    docker run --runtime=nvidia -ti --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --mount type=bind,source=$(pwd),target=/app\
    --mount type=bind,source=/data/,target=/data\
    criteo_preprocessing:latest  bash

In our example the original tsv files are located in /data/criteo_tb.

To convert the dataset we then execute the following command:

    bash preproc.sh /data/criteo_tb /data

The final result can then be found in /data/in_mem.
