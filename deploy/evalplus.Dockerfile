ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG EVALPLUS_REVISION
RUN test -n "${EVALPLUS_REVISION}"
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/evalplus/evalplus.git /opt/evalplus \
    && git -C /opt/evalplus checkout --detach "${EVALPLUS_REVISION}" \
    && test "$(git -C /opt/evalplus rev-parse HEAD)" = "${EVALPLUS_REVISION}" \
    && pip install --no-cache-dir /opt/evalplus

LABEL org.opencontainers.image.source="https://github.com/evalplus/evalplus"
LABEL org.opencontainers.image.revision="${EVALPLUS_REVISION}"
WORKDIR /work
