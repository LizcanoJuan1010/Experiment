FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies + Python 3.10
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    vim \
    wget \
    unzip \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.6+ with CUDA 12.8 support (required for RTX 5060 Ti / Blackwell)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu128

# Install python dependencies
RUN pip install --no-cache-dir \
    transformers>=4.36.0 \
    accelerate \
    bitsandbytes \
    pyvene \
    Pillow \
    pycocotools \
    scikit-learn \
    pandas \
    scipy \
    statsmodels \
    matplotlib \
    seaborn \
    tqdm \
    datasets \
    huggingface_hub

# Model weights are NOT baked into the image.
# They are downloaded at first run and cached on the host via volume mount:
#   -v ${PWD}\hf_cache:/root/.cache/huggingface

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Copy project files
COPY . /app/

CMD ["bash"]
