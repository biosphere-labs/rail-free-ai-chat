FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY app.py .
COPY chainlit.md .
COPY agent/ agent/
COPY tts/ tts/

# Install dependencies
RUN uv sync --no-dev

# Expose Chainlit port
EXPOSE 8000

# Create mount point for host home directory
RUN mkdir -p /host_home

# Run Chainlit
CMD ["uv", "run", "chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
