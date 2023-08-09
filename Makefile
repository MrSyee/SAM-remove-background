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


# k8s
cluster:
	curl -sfL https://get.k3s.io | INSTALL_K3S_VERSION="v1.27.2+k3s1" K3S_KUBECONFIG_MODE="644" INSTALL_K3S_EXEC="server --disable=traefik" sh -s - --docker
	mkdir -p ~/.kube
	cp /etc/rancher/k3s/k3s.yaml ~/.kube/config

	kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/nvidia-device-plugin.yml
	kubectl create -f https://raw.githubusercontent.com/NVIDIA/dcgm-exporter/master/dcgm-exporter.yaml

	helm repo add minio https://helm.min.io/
	helm repo add grafana https://grafana.github.io/helm-charts
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
	helm repo update

finalize:
	sh /usr/local/bin/k3s-killall.sh
	sh /usr/local/bin/k3s-uninstall.sh

charts:
	kubectl apply -f secrets/triton.yaml
	helm install minio charts/minio
	helm install triton charts/triton
