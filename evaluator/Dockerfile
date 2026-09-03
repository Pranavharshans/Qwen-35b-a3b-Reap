ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# BASE_IMAGE must be a digest-pinned image containing python3, javac, and java.
RUN test -x /usr/bin/python3 && test -x /usr/bin/javac && test -x /usr/bin/java
USER 65534:65534
