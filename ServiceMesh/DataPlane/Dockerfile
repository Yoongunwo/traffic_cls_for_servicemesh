FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

RUN apt-get update && apt-get install -y \
    git \
    python3.10 \
    python3.10-distutils \
    python3.10-dev \
    python3-pip \
    python3-setuptools \
    net-tools \
    procps \
    iptables-persistent \
    sudo \
    curl \
    vim \
    wget \
    iputils-ping \
    net-tools \
    joblib \
    scikit-learn \
    uvloop

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir wheel setuptools
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install uvloop
RUN python3 -m pip install joblib

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# pip 업그레이드
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./ ./
RUN pip install -r requirements.txt

# 새 사용자 생성 및 sudo 권한 부여 
RUN useradd -m -s /bin/bash -u 5555 proxyuser && \
    echo "proxyuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# 앱 디렉토리 권한 설정
RUN chown -R proxyuser:proxyuser /app

# 실행 권한 부여
RUN chmod +x /app/iptables.sh

# model folder
RUN mkdir /app/model
RUN chmod -R 777 /app/model

RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu

ENTRYPOINT ["/bin/sh", "-c", "/app/iptables.sh && while true; do sleep 1000; done"]