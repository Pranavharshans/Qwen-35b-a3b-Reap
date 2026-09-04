# Base image for the frozen evaluator/Dockerfile, which requires a
# digest-pinned BASE_IMAGE containing python3, javac, and java.
# BASE_IMAGE must be passed as a digest pin (name@sha256:...); the prepare
# script resolves and records it at build time — never a floating tag.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 openjdk-21-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

RUN test -x /usr/bin/python3 && test -x /usr/bin/javac && test -x /usr/bin/java
