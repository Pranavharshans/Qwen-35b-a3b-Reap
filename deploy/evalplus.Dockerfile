ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG EVALPLUS_REVISION
RUN test -n "${EVALPLUS_REVISION}"
# The pinned SHA-256 belongs to the official HumanEval+ v0.1.9 release asset
# (verified against the release download). v0.1.10 is a different archive, so
# version and hash must stay paired.
ARG HUMANEVAL_PLUS_VERSION=v0.1.9
ARG HUMANEVAL_PLUS_SHA256=e62f4130146963d969da64553f407a66e52d095adbfed4ee6733b4d59e14a3ed
ENV HUMANEVAL_PLUS_PATH=/opt/evalplus-data/HumanEvalPlus.jsonl.gz
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/evalplus/evalplus.git /opt/evalplus \
    && git -C /opt/evalplus checkout --detach "${EVALPLUS_REVISION}" \
    && test "$(git -C /opt/evalplus rev-parse HEAD)" = "${EVALPLUS_REVISION}" \
    && pip install --no-cache-dir /opt/evalplus

RUN mkdir -p /opt/evalplus-data
RUN python -c "import hashlib, pathlib, urllib.request; p=pathlib.Path('/opt/evalplus-data/HumanEvalPlus.jsonl.gz'); urllib.request.urlretrieve('https://github.com/evalplus/humanevalplus_release/releases/download/${HUMANEVAL_PLUS_VERSION}/HumanEvalPlus.jsonl.gz', p); assert hashlib.sha256(p.read_bytes()).hexdigest() == '${HUMANEVAL_PLUS_SHA256}'" \
    && test "$(sha256sum /opt/evalplus-data/HumanEvalPlus.jsonl.gz | cut -d ' ' -f1)" = "${HUMANEVAL_PLUS_SHA256}"

LABEL org.opencontainers.image.source="https://github.com/evalplus/evalplus"
LABEL org.opencontainers.image.revision="${EVALPLUS_REVISION}"
LABEL org.opencontainers.image.humanevalplus.version="${HUMANEVAL_PLUS_VERSION}"
LABEL org.opencontainers.image.humanevalplus.path="/opt/evalplus-data/HumanEvalPlus.jsonl.gz"
LABEL org.opencontainers.image.humanevalplus.sha256="${HUMANEVAL_PLUS_SHA256}"
WORKDIR /work
