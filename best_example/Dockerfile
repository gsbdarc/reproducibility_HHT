# Start from a CUDA-enabled base image
FROM docker.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Install Python and other necessary packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Set the Python alias for convenience
RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy the requirements file and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script into the container
COPY . /app

# Define the command to run the script
CMD ["python", "app.py"]

