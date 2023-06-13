FROM python:3.9-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 git gcc make

WORKDIR /app
COPY src/server/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY logging.conf .
COPY scripts/run_server.sh .
COPY src/server src/server

COPY assets assets/

EXPOSE 8888

CMD [ "sh", "run_server.sh" ]
