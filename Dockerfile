FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-nld \
    libtesseract-dev \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libopencv-dev \
    python3-opencv \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Tesseract combined models with both legacy and LSTM components for OEM 2 support
RUN wget -O /usr/share/tesseract-ocr/5/tessdata/eng.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata \
    && wget -O /usr/share/tesseract-ocr/5/tessdata/nld.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/nld.traineddata

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m spacy download nl_core_news_sm
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m spacy download en_core_web_sm

# Copy project files
COPY . .

# Copy and set entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]

# Default command
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
