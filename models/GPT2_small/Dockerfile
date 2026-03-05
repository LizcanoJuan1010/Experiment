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

# Set environment variables
ENV PYTHONPATH=/app

# Copy project files
COPY . /app/

CMD ["bash"]
