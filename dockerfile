# Small, fast image; works on Unraid (x86_64)
FROM python:3.10-slim

# Streamlit defaults + bigger uploads
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXsrfProtection=false \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200

# SciPy wheels don’t need compilers; libgomp is tiny and helps on some hosts
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Add the app
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "final.py"]
