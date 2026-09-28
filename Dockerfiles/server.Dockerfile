FROM python:3.9.25-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 gcc make

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra server

ENV PATH="/app/.venv/bin:$PATH"

COPY logging.conf .
COPY scripts/run_server.sh .
COPY src/server src/server

COPY assets assets/

EXPOSE 8888

CMD [ "sh", "run_server.sh" ]
