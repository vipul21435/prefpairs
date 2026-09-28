# syntax=docker/dockerfile:1
# PrefPairs CLI image: a uv-built virtualenv on a digest-pinned Python 3.12 slim
# base, run as a non-root user. Build and try it with:
#   docker build -t prefpairs .
#   docker run --rm prefpairs simulate --seed 0
#   docker run --rm --entrypoint sh prefpairs scripts/demo.sh

ARG PYTHON_IMAGE=python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.29@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
# Dependencies first, so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM ${PYTHON_IMAGE}
LABEL project=prefpairs \
      org.opencontainers.image.title="prefpairs" \
      org.opencontainers.image.description="RLHF preference-data toolkit CLI" \
      org.opencontainers.image.source="https://github.com/vipul21435/prefpairs" \
      org.opencontainers.image.licenses="MIT"
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin prefpairs \
    && mkdir /data \
    && chown prefpairs:prefpairs /data
COPY --from=build /opt/venv /opt/venv
COPY examples /app/examples
COPY scripts /app/scripts
ENV PATH=/opt/venv/bin:$PATH \
    DEMO_DB=/data/demo.db \
    DEMO_EXPORT_DIR=/data/demo-export \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PREFPAIRS_DB=/data/prefpairs.db
WORKDIR /app
USER prefpairs
VOLUME ["/data"]
ENTRYPOINT ["prefpairs"]
CMD ["--help"]
