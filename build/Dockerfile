FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HARADIBOTS_CACHE_ROOT=/var/lib/haradibots/cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN mkdir -p /var/lib/haradibots/cache \
    && useradd --system --uid 10001 --create-home haradibots \
    && chown -R haradibots:haradibots /app /var/lib/haradibots

USER haradibots
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["python3.11", "-m", "uvicorn", "cluster.api_server:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
