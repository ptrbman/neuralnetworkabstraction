# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /workspace
WORKDIR /workspace

# Install PyTorch, TensorFlow
RUN pip install --no-cache-dir torch torchvision torchaudio tensorflow maraboupy

# Install additional useful packages
RUN pip install --no-cache-dir numpy pandas matplotlib ipython jupyterlab scikit-learn

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Configure environment
ENV PYTHONUNBUFFERED=1
