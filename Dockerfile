FROM python:3.11-slim

WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required directories
RUN mkdir -p data/processed models/embeddings models/ranker results

# Expose Streamlit and FastAPI ports
EXPOSE 8501 8000

# Default command — run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# To run FastAPI instead:
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
