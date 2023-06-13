FROM python:3.9-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

WORKDIR /app
COPY src/client/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/client/app.py .
COPY src/client/utils.py .
COPY checkpoint/sam_onnx_quantized.onnx checkpoint/
COPY assets assets/

EXPOSE 7860

CMD [ "python", "app.py" ]
