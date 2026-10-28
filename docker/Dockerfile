# The Lernex engine as a container: `lernex serve` on port 8080, with the CUDA build (the same binary falls back to
# the CPU on a machine without a GPU). Build from this directory:
#
#   docker build -t ghcr.io/lernex/lernex --build-arg LERNEX_VERSION=0.1.0 -f docker/Dockerfile docker
#
# Run it with the GPU (needs the NVIDIA Container Toolkit on the host):
#
#   docker run --gpus all -p 8080:8080 -v lernex:/root/.cache/lernex ghcr.io/lernex/lernex praxis
#
# The image takes the release tarball from the release host, checks it against SHA256SUMS, and unpacks it to
# /opt/lernex, the tree install.sh builds on. The base is NVIDIA's CUDA base for the CUDA major the tarball bundles
# (its libcudart and libcublasLt come with the tarball, so the base needs no toolkit): the smallest one with the
# NVIDIA runtime hooks. Arguments:
#   LERNEX_VERSION   the release, X.Y.Z (required)
#   LERNEX_FLAVOR    the build: cuda12 (the CUDA build a release ships)
#   LERNEX_RELEASES  the release host, laid out like GitHub's
#   CUDA_BASE        the base, NVIDIA's CUDA base for the tarball's toolkit (12.8 for cuda12)

ARG LERNEX_VERSION
ARG LERNEX_FLAVOR=cuda12
ARG LERNEX_RELEASES=https://github.com/lernex/lernex-engine/releases
ARG CUDA_BASE=nvidia/cuda:12.8.1-base-ubuntu24.04

# Fetch and check the tree on a plain Ubuntu: the runtime image gets the tree, not the download tooling.
FROM ubuntu:24.04 AS fetch
ARG LERNEX_VERSION
ARG LERNEX_FLAVOR
ARG LERNEX_RELEASES
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
RUN set -eux; \
    test -n "${LERNEX_VERSION}"; \
    base="${LERNEX_RELEASES%/}/download/v${LERNEX_VERSION}"; \
    asset="lernex-linux-x86_64-${LERNEX_FLAVOR}.tar.gz"; \
    mkdir -p /tmp/dl /opt/lernex; \
    cd /tmp/dl; \
    curl -fsSL --retry 3 -o SHA256SUMS "${base}/SHA256SUMS"; \
    curl -fsSL --retry 3 -o "${asset}" "${base}/${asset}"; \
    grep " ${asset}\$" SHA256SUMS | sha256sum -c -; \
    tar -xzf "${asset}" -C /opt/lernex; \
    rm -rf /tmp/dl; \
    /opt/lernex/bin/lernex --version

FROM ${CUDA_BASE}
ARG LERNEX_VERSION
LABEL org.opencontainers.image.title="Lernex engine" \
      org.opencontainers.image.description="The Lernex inference engine for Praxis and Logos, serving the OpenAI API on port 8080" \
      org.opencontainers.image.source="https://github.com/lernex/lernex-engine" \
      org.opencontainers.image.version="${LERNEX_VERSION}" \
      org.opencontainers.image.licenses="NOASSERTION"
# The NVIDIA runtime hooks need these to hand the GPU to the container. curl brings the libcurl.so.4 the engine
# opens to download models (the package's name differs by Ubuntu release; the package manager picks it).
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=fetch /opt/lernex /opt/lernex
RUN ln -s /opt/lernex/bin/lernex /usr/local/bin/lernex \
 && echo docker > /opt/lernex/share/lernex/channel
VOLUME /root/.cache/lernex
EXPOSE 8080
# The server binds every interface and port 8080 inside the container: a bare model name is all a run needs, and
# any other flags follow it (docker run ... praxis --context 8192). Override the entrypoint for the other commands.
ENTRYPOINT ["lernex", "serve", "--host", "0.0.0.0", "--port", "8080"]
CMD ["praxis"]
