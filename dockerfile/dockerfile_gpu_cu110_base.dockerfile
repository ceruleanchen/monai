FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

ENV DISPLAY :10
ENV DEBIAN_FRONTEND=noninteractive

# Install apt package
RUN apt-get update && \
    apt-get install -y sudo vim git wget curl zip unzip && \
    apt-get install -y net-tools iputils-ping apt-utils && \
    apt-get install -y build-essential && \
    # For opencv (ImportError: libGL.so.1)
    apt-get install -y libgl1-mesa-glx
    # # ImportError: libSM.so.6
    # apt-get install -y libsm6 libxrender-dev libxext6 && \
    # # Failed building wheel for Pillow
    # apt-get install -y libjpeg-dev zlib1g-dev

# Install python3.7
RUN apt-get install -y software-properties-common && \
    apt-get install -y python3-pip python3.7-dev && \
    cd /usr/bin && \
    rm python3 && \
    ln -s python3.7 python && \
    ln -s python3.7 python3 && \
    ln -s pip3 pip && \
    pip install --upgrade pip

# Install python3.7 package
RUN pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html  && \
    pip3 install opencv-python && \
    pip3 install matplotlib && \
    pip3 install pandas && \
    pip3 install sklearn && \
    pip3 install scikit-image && \
    pip3 install tensorboard

# https://github.com/Project-MONAI/tutorials
COPY requirements-dev.txt requirements-min.txt requirements.txt ./
RUN pip3 install monai && \
    # pip3 install -r https://raw.githubusercontent.com/Project-MONAI/MONAI/dev/requirements-dev.txt && \
    pip3 install -r requirements-dev.txt && \
    rm requirements-dev.txt requirements-min.txt requirements.txt && \
    pip3 install dicom2nifti && \
    pip3 install sanic && \
    pip3 install boto3
