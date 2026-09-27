# Dockerfile — RAG Agent application container.
#
# Runs the Streamlit web UI on port 8501.
# Ollama runs as a separate service (see docker-compose.yml).
#
# Build:
#   docker build -t rag-agent .
#
# Run standalone (expects Ollama on the host):
#   docker run -p 8501:8501 \
#     -e OLLAMA_HOST=http://host.docker.internal:11434 \
#     -v $(pwd)/chroma_db:/app/chroma_db \
#     -v $(pwd)/docs:/app/docs \
#     rag-agent
#
# For the full stack (app + Ollama together), use docker-compose.yml instead.

# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.11 slim keeps the image small while matching the required version.
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential  — compiles Python packages that have C extensions (chromadb, lxml)
# curl             — used in the health check to verify Ollama is reachable
# git              — some pip packages fetch metadata via git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
 && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer.
# The layer is only re-built when requirements.txt changes — not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY . .

# ── Runtime configuration ─────────────────────────────────────────────────────
# Port 8501 is the Streamlit default.
EXPOSE 8501

# RAG_OLLAMA_HOST lets docker-compose override where Ollama is reachable.
# Defaults to localhost — overridden to the ollama service in docker-compose.yml.
ENV RAG_OLLAMA_HOST=http://localhost:11434

# Streamlit configuration — disable the browser auto-open and the usage stats prompt
# so the container starts cleanly without interactive prompts.
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true

# ── Entry point ───────────────────────────────────────────────────────────────
# --server.address=0.0.0.0 makes Streamlit reachable from outside the container.
# --server.port=8501 is explicit to match the EXPOSE declaration above.
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501"]
