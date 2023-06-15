PYTHON=3.9
BASENAME=sam-remove-background
CONTAINER_NAME=ghcr.io/mrsyee/${BASENAME}

# Prerequisites for local execution
env:
	conda create -n $(BASENAME)  python=$(PYTHON) -y

setup:
	pip install -r requirements.txt

model:
	sh scripts/download_decoder_model.sh
	sh scripts/download_encoder_model.sh

# Run
client:
	python src/client/app.py

server:
	PYTHONPATH=src/server uvicorn src.server.main:app --host 0.0.0.0 --port 8888

server-debug:
	PYTHONPATH=src/server uvicorn src.server.main:app --host 0.0.0.0 --port 8888 --reload --reload-exclude "src/client/*"

# Docker
# Client
docker-build-client:
	docker build -t ${CONTAINER_NAME}-client -f Dockerfiles/client.Dockerfile .

docker-push-client:
	docker push ${CONTAINER_NAME}-client

docker-pull-client:
	docker pull ${CONTAINER_NAME}-client

# Server
docker-build-server:
	docker build -t ${CONTAINER_NAME}-server -f Dockerfiles/server.Dockerfile .

# Dev
setup-dev:
	pip install -r requirements-dev.txt

format:
	black .
	isort .

lint:
	PYTHONPATH=src pytest src --flake8 --pylint --mypy