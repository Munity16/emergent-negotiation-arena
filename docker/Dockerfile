# ─────────────────────────────────────────────────────────
# Emergent Negotiation Arena
#
#   docker build -f docker/Dockerfile -t emergent-arena .
#
# LLM agents via Fireworks AI:
#   docker run -e FIREWORKS_API_KEY=your_key \
#     -e AGENT_BACKEND=fireworks \
#     -p 7860:7860 \
#     emergent-arena
#
# No API key (rule-based demo / recorded replay):
#   docker run -e AGENT_BACKEND=heuristic -p 7860:7860 emergent-arena
#   docker run -e AGENT_BACKEND=replay    -p 7860:7860 emergent-arena
# ─────────────────────────────────────────────────────────

# Stage 1: Build deps
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Runtime
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Emergent Negotiation Arena"
LABEL org.opencontainers.image.description="Emergent symbolic negotiation between LLM agents — Fireworks AI / no-key demo backends"

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy project
COPY . .

# Create outputs directory
RUN mkdir -p /app/outputs

# Expose Gradio port
EXPOSE 7860

# Environment defaults (override at runtime)
ENV AGENT_BACKEND=auto
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Health check — python:3.11-slim ships no curl, so probe with the stdlib
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/', timeout=5)" || exit 1

# Default: launch the Gradio UI
CMD ["python", "main.py", "--ui"]
