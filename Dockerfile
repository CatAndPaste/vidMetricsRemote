FROM nvidia/cuda:11.8.0-cudnn8.6-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    ln -fs /usr/share/zoneinfo/Europe/Moscow /etc/localtime && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get install -y \
    cmake \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk2.0-dev \
    libboost-python-dev \
    libboost-thread-dev \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libgl1-mesa-glx \
    wget && \
    apt-get clean

WORKDIR /tmp
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh && \
    bash Anaconda3-2021.05-Linux-x86_64.sh -b -p /opt/conda && \
    rm Anaconda3-2021.05-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"

COPY environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml

COPY . /workspace

WORKDIR /workspace

CMD ["conda", "run", "--no-capture-output", "-n", "vidMetrics", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
