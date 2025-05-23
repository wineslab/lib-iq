FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        apt-utils \
        software-properties-common \
        wget \
        git \
        cmake \
        g++ \
        libtool \
        graphviz \
        swig \
        python3-pip

RUN python3 -m pip install --upgrade \
    pip \
    setuptools \
    wheel \
    pytest \
    build \
    hatch

RUN git clone -b main https://github.com/wineslab/lib-iq /libiq

WORKDIR /libiq

RUN ./build.sh

RUN pip install dist/libiq-*.tar.gz

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

CMD [ "/bin/sh" ]
