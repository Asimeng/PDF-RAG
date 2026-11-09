# Container image for the RAG API.
#   docker build -t pdf-rag .
#   docker run -p 8000:8000 -v ./data:/app/data -e RAG_OLLAMA_URL=http://host.docker.internal:11434 pdf-rag
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAG_API_HOST=0.0.0.0 \
    RAG_DATA_DIR=/app/data

WORKDIR /app

# libglib is needed by OpenCV, which the OCR engine uses.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (cached layer), then the source.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

# Optional: GPU acceleration for embeddings/reranker inside the container.
ARG WITH_CUDA=false
RUN if [ "$WITH_CUDA" = "true" ]; then pip uninstall -y onnxruntime && pip install "onnxruntime-gpu[cuda,cudnn]"; fi

# Download the OCR models now (~15 MB), so containers also work without internet.
RUN python -c "from rapidocr import RapidOCR; RapidOCR(params={'Global.log_level': 'warning'})"

COPY eval ./eval

RUN useradd --create-home appuser && mkdir -p /app/data && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["rag", "serve"]
