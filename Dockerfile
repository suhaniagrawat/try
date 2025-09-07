# Use Python 3.11 with OpenCV support
FROM python:3.11-slim

# Install system dependencies for OpenCV and AI libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libgtk-3-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy frontend package files and install dependencies
COPY frontend/package*.json ./frontend/
WORKDIR /app/frontend
RUN npm install

# Copy frontend source and build
COPY frontend/ ./
RUN npm run build

# Switch back to app root
WORKDIR /app

# Copy backend source
COPY backend/ ./backend/

# Copy built frontend to backend's static directory
RUN mkdir -p backend/static && cp -r frontend/dist/* backend/static/

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app/backend
ENV NODE_ENV=production
ENV HEADLESS=true
ENV RAILWAY_ENVIRONMENT=production

# Start command (backend only, AI agent separate)
CMD ["python", "backend/main.py"]
