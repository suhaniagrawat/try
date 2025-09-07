# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Ensures Python output is sent straight to the terminal without buffering
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY backend/requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir: Disables the cache, which reduces the image size.
# --upgrade pip: Ensures we have the latest version of pip.
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# Copy the entire backend directory into the container at /app/backend
COPY ./backend /app/backend

# The command to run your application when the container launches.
# This starts the Uvicorn server for your FastAPI application.
# The PYTHONPATH is set so that Python can find your modules inside the /app/backend directory.
CMD ["sh", "-c", "PYTHONPATH=/app/backend uvicorn main:app --host 0.0.0.0 --port $PORT"]
