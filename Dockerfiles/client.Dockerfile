FROM python:3.9-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

WORKDIR /app
COPY src/client/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/client .

COPY checkpoint/sam_onnx_quantized.onnx checkpoint/
ENV CHECKPOINT_PATH "checkpoint/sam_onnx_quantized.onnx"

COPY assets assets/
ENV EXAMPLE_DIR "assets/examples"


EXPOSE 7860

CMD [ "python", "app.py" ]
