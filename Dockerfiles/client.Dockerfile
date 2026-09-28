FROM python:3.9.25-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra client

ENV PATH="/app/.venv/bin:$PATH"

COPY src/client .

COPY checkpoint/sam_onnx_quantized.onnx checkpoint/
ENV CHECKPOINT_PATH="checkpoint/sam_onnx_quantized.onnx"

COPY assets assets/
ENV EXAMPLE_DIR="assets/examples"


EXPOSE 7860

CMD [ "python", "app.py" ]
