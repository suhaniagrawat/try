# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy all dependency and code files
COPY . .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on (Railway will map this)
EXPOSE 8000