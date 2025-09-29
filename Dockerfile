FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen

COPY . .

CMD ["python", "tfx_pipeline/pipeline.py"]

