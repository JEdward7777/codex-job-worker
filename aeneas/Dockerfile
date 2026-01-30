FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    python2.7 \
    python-pip \
    python-dev \
    build-essential \
    ffmpeg \
    espeak \
    libespeak-dev \
    libsndfile1 \
    sox \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (last version that supports py2)
RUN pip install --upgrade "pip<21"

# Install aeneas
RUN pip install numpy==1.16.6
RUN pip install aeneas

# Set working directory
WORKDIR /work

# Default command just shows help
CMD ["python", "-m", "aeneas.tools.execute_task", "--help"]

#building it with
#docker build -t aeneas-py27 .