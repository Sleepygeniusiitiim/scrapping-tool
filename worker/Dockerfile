# Background worker: persistent process with headless Chromium (Playwright), no request time limit.
# Build from the repository root:   docker build -f worker/Dockerfile -t sourcing-worker .
FROM mcr.microsoft.com/playwright/python:v1.49.1-noble

WORKDIR /app
COPY requirements.txt ./requirements.txt
COPY worker/requirements.txt ./worker/requirements.txt
RUN pip install --no-cache-dir -r worker/requirements.txt
COPY . .

ENV BROWSER_FETCH=1 \
    WORKER_SCALE=3 \
    RUNTIME=worker \
    PYTHONUNBUFFERED=1

CMD ["python", "worker/worker.py"]
