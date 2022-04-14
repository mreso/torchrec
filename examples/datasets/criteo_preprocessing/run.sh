#!/bin/bash

docker build -t criteo_preprocessing:latest . && \
	docker run --runtime=nvidia -ti --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
	--mount type=bind,source=$(pwd),target=/app\
	--mount type=bind,source=/data/,target=/data\
	--mount type=bind,source=/ssd/,target=/ssd\
	 criteo_preprocessing:latest  bash preproc.sh /data/criteo_tb /ssd/criteo_preproc
