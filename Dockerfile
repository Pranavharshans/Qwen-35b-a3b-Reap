ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG UV_VERSION=0.8.15
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/opt/uv-cache \
    HF_HOME=/models/huggingface

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"
WORKDIR /workspace
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --extra gpu --no-dev
COPY configs ./configs
COPY scripts ./scripts

ENTRYPOINT ["uv", "run", "--frozen", "--no-sync", "reverse-reap"]
