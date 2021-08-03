FROM nvidia/cuda:11.4.0-devel-ubuntu20.04
#10.1-cudnn7-devel-ubuntu18.04

ENV DCUDA_ARCHS=""86;86""

# Install tools and dependencies.
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

RUN DEBIAN_FRONTEND="noninteractive"  apt-get -y update --fix-missing
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y \
	emacs \
	git \
	wget \
	libgoogle-glog-dev

# Install TensorFlow and pytorch.
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y python3-dev python3-pip
RUN pip3 install --upgrade pip && pip install --upgrade tensorflow_gpu
RUN pip install --upgrade tensor2tensor && pip install tqdm
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

#Install Pybind
RUN pip install pybind11 && pip install "pybind11[global]"

#Rename python
RUN apt-get install python-is-python3

# Install CMake.
RUN DEBIAN_FRONTEND="noninteractive"  apt-get install -y software-properties-common && \
    apt-get update && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y cmake

# Install Sputnik.
RUN mkdir /mount
WORKDIR /mount
RUN git clone --recursive https://github.com/google-research/sputnik.git && \
	mkdir sputnik/build
WORKDIR /mount/sputnik/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=OFF -DBUILD_BENCHMARK=OFF \
	-DCUDA_ARCHS=$DCUDA_ARCHS -DCMAKE_INSTALL_PREFIX=/usr/local/sputnik && \
	make -j8 install

# Copy the source into the image.
RUN mkdir -p /mount/sgk
COPY ./sgk /mount/sgk/

# Install SGK.
RUN mkdir /mount/sgk/build
WORKDIR /mount/sgk/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS=$DCUDA_ARCHS \
	-DCMAKE_INSTALL_PREFIX=/usr/local/sgk && \
	make -j8 install

# Setup the environment.
ENV PYTHONPATH="/mount:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="`python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())'`:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="/usr/local/sputnik/lib:${LD_LIBRARY_PATH}"

#Setup cudnn
RUN sh /mount/sgk/cudnn.sh

#Setup Cusparse/Cublas
RUN mkdir -p /mount/_c
COPY ./_c /mount/_c
WORKDIR /mount/_c
RUN make all

# Set the working directory.
WORKDIR /mount
