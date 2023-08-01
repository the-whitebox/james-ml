# Use the official Python base image for ARM
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0

# Copy the padim_requirements.txt file to the container
COPY padim_requirements.txt .

# Install Python dependencies
RUN pip install -r padim_requirements.txt

# Install PyTorch
# RUN pip install --no-cache-dir torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Copy the application files to the container
COPY . .

# Expose the port that the application listens on
EXPOSE 5012

# Set the entrypoint command to run the application
# CMD ["python", "lightning_inference.py"]
ENTRYPOINT ["python"]

CMD ["lightning_inference.py"]