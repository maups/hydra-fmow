# team: usf_bulls
# This is a GPU-based system that uses fmow-rgb sucessfully tested with cuda 8.0 and cudnn 6
# Our test script removes temporary files and predictions that were previously generated
# Our train script removes temporary files and models that were previously generated

# ARG cuda_version=8.0
# ARG cudnn_version=6
# FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel
FROM nvidia/cuda:8.0-cudnn6-devel

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y vim bc curl wget git libhdf5-dev g++ graphviz openmpi-bin libgl1-mesa-glx && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh

ENV NB_USER keras
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown keras $CONDA_DIR -R && \
    mkdir -p /src && \
    chown keras /src

COPY . /src/

RUN chown keras /src -R && \
    chmod u+x /src/train.sh /src/test.sh && \
    g++ /src/iarpa/fusion.cpp -o /src/iarpa/fusion -std=c++11

USER keras

RUN conda install -y python=2.7 && \
    pip install --upgrade pip && \
    pip install tensorflow-gpu && \
    pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0rc1-cp27-none-linux_x86_64.whl && \
    conda install Pillow scikit-learn notebook pandas matplotlib mkl nose pyyaml six h5py && \
    conda install pygpu bcolz && \
    pip install sklearn_pandas && \
    pip install git+git://github.com/keras-team/keras.git && \
    pip install tqdm && \
    conda clean -yt

ENV PYTHONPATH /src/:$PYTHONPATH
ENV KERAS_BACKEND tensorflow

WORKDIR /src

