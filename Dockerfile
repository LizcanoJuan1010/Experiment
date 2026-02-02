FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Fix torchvision compatibility with newer transformers
RUN pip uninstall -y torchvision torchaudio || true

# Install python dependencies
RUN pip install --no-cache-dir \
    transformer_lens \
    einops \
    jaxtyping \
    scikit-learn \
    pandas \
    jupyter \
    accelerate \
    transformers \
    statsmodels \
    scipy \
    tqdm \
    datasets

# Set environment variables
ENV PYTHONPATH=/app

# Copy project files
COPY . /app/

CMD ["bash"]
