# --- services/docker-router/Dockerfile ---
FROM docker:20.10

RUN apk add --no-cache python3 py3-pip

WORKDIR /app
# https://github.com/docker/docker-py/issues/3256
RUN pip install fastapi==0.111.0 uvicorn==0.29.0 docker==7.0.0 pydantic==2.7.1 requests==2.31.0
RUN pip install langchain==0.3.18 langchain-nvidia-ai-endpoints==0.3.9

# Copy application
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8070/health || exit 1

ENTRYPOINT [ -f assessment/run_assessment.py ] && python3 assessment/run_assessment.py; \
    uvicorn docker_router:app --host 0.0.0.0 --port 8070