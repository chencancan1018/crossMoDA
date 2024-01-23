
FROM nvidia/cuda:11.0.3-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.7 \
    python3-pip \
    python3-setuptools \
    build-essential \
    cmake \
    protobuf-compiler \
    libgtk2.0-dev \
    libprotoc-dev \
    libopenblas-dev \
    liblapack-dev \
    libgl1 \
    curl \
    tzdata \
    && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.7 /usr/bin/python3

COPY  src/requirements1.txt .
COPY  src/requirements2.txt .
COPY  src/input .

RUN python3 -V

RUN python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip setuptools wheel

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ 

RUN python3 -m pip install -r requirements1.txt

RUN python3 -m pip install -r requirements2.txt

COPY ./ ./

CMD ["/src/run_inference.py"]
ENTRYPOINT ["python3"]