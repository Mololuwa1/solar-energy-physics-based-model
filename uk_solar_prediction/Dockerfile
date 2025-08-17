# Dockerfile for UK Solar Energy Prediction Model
# Optimized for AWS SageMaker deployment with physics-informed neural networks

FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /opt/ml/code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY sagemaker/ ./

# Set environment variables for SageMaker
ENV PYTHONPATH="/opt/ml/code/src:${PYTHONPATH}"
ENV SAGEMAKER_PROGRAM=train.py

# Create necessary directories
RUN mkdir -p /opt/ml/input/data/train \
    && mkdir -p /opt/ml/input/data/validation \
    && mkdir -p /opt/ml/input/data/test \
    && mkdir -p /opt/ml/model \
    && mkdir -p /opt/ml/output \
    && mkdir -p /opt/ml/checkpoints

# Set permissions
RUN chmod +x train.py inference.py

# Default command for training
CMD ["python", "train.py"]

