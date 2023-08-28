FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

ARG PACKAGES="ffmpeg build-essential"
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends -qq $PACKAGES && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN apt-get update && apt-get install -y python3.8 python3-pip

# Install requirements
ADD requirements.txt .
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

# Copy the model to the container
COPY model.pt /opt/ml/model/model.pt
COPY code/ /opt/ml/model/code

# Defining container entrypoint
ENTRYPOINT ["python", "/opt/ml/model/code/inference.py"]