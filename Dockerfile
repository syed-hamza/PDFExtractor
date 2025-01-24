# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Install Java and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p uploads storage/indices && \
    chmod 755 uploads storage storage/indices

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Run the application
CMD ["python", "app.py"]