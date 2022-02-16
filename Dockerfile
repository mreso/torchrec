FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install -y wget vim #&& rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda install -y pytorch cudatoolkit=11.3 -c pytorch-nightly \
    && conda install -y numpy torchmetrics iopath scikit-build jinja2 ninja cmake git -c conda-forge

RUN git clone --recursive https://github.com/pytorch/FBGEMM \
    && cd FBGEMM/fbgemm_gpu/ \
    && cp /usr/include/x86_64-linux-gnu/cudnn_v8.h /usr/include/x86_64-linux-gnu/cudnn.h \
    && cp /usr/include/x86_64-linux-gnu/cudnn_version_v8.h /usr/include/x86_64-linux-gnu/cudnn_version.h \
    && python setup.py install -DCUDNN_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/libcudnn.so -DCUDNN_INCLUDE_PATH=/usr/include/x86_64-linux-gnu/

RUN git clone --recursive https://github.com/facebookresearch/torchrec \
    && cd torchrec/ 
WORKDIR /torchrec
RUN git checkout 932d9bb5b4bf765e4e238184774d46b745a65ad6
RUN python setup.py build develop --skip_fbgemm

RUN pip install torchx-nightly

RUN cd / \
    && git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp \
    && cd aws-sdk-cpp/ \
    && mkdir sdk-build \
    && cd sdk-build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY="s3;transfer" \
    && make \
    && make install 

RUN conda install -y pybind11 -c conda-forge \
    && apt install -y libz-dev libcurl4-openssl-dev libssh-dev \
    && cd / \
    && git clone https://github.com/ydaiming/data.git \
    && cd data/ \
    && git checkout s3-datapipes \
    && git checkout b1e00513a25380b43d7e0aa5c3f521519e2c953a \
    && export BUILD_S3=1 \
    && python setup.py build \
    && python setup.py install


COPY credentials /root/.aws/credentials

