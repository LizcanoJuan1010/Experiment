FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies + Python 3.10
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    vim \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.6+ with CUDA 12.8 support (required for RTX 5060 Ti / Blackwell)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu128

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
    datasets \
    nltk \
    wordfreq \
    spacy \
    matplotlib \
    seaborn

# Download NLTK data (needed for corpus extraction mode)
RUN python -c "\
import nltk; \
nltk.download('brown', quiet=True); \
nltk.download('gutenberg', quiet=True); \
nltk.download('punkt', quiet=True); \
nltk.download('punkt_tab', quiet=True); \
nltk.download('wordnet', quiet=True); \
nltk.download('averaged_perceptron_tagger', quiet=True); \
nltk.download('averaged_perceptron_tagger_eng', quiet=True)"

# Download spaCy English model (needed for dependency parsing)
RUN python -m spacy download en_core_web_sm

# Pre-download Pythia 2.8B model weights (cached in image, ~5.6GB float16).
# This avoids re-downloading on every container run.
# Comment out this line if you prefer to download at runtime (smaller image).
RUN python -c "from transformer_lens import HookedTransformer; import torch; \
    HookedTransformer.from_pretrained('pythia-2.8b', dtype=torch.float16)"

# Set environment variables
ENV PYTHONPATH=/app
# Efficient CUDA memory allocation for large models
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Copy project files
COPY . /app/

CMD ["bash"]
