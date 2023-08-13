# SAM Extract Object & Remove background App
[Segment Anything Model(SAM)](https://github.com/facebookresearch/segment-anything) is the foundation model for the segmentation task by Meta.
In this repository, an application that extract an object and remove the background by utilizing SAM's interactive segmentation with click is implemented. The UI is implemented using [gradio](https://gradio.app/).
The Encoder part of SAM is served to the [triton inference server](https://github.com/triton-inference-server/server) to increase the efficiency of inference.
All parts of the app were configured for deployment on docker and k8s.

Demo video([Youtube](https://youtu.be/R3BP1GmKroA)):

https://github.com/MrSyee/SAM-remove-background/assets/17582508/1ae9ffb8-47ea-4025-afa5-e158ab209665

## Prerequisite
- python 3.9+
- [docker](https://www.docker.com/)
- GPU: Required for the speed of the image encoder

## Contents
- [x] Implement remove background app using SAM with Gradio.
- [x] Docker compose for the server and client.
- [x] Convert pre-trained SAM Encoder to torchscript. ([Huggingface](https://huggingface.co/khsyee/sam-vit-h-encoder-torchscript/tree/main))
- [x] Triton serving SAM Encoder.
- [x] Kubernetes helm charts for the server and client.
- [x] Monitoring on K8s (Promtail + Loki & Prometheus & Grafana).

## Application Structure
![app-structure](https://github.com/MrSyee/SAM-remove-background/assets/17582508/97ac20d0-a083-499b-bdde-5a3e8b5c662a)

SAM has three components: **an image encoder**, **a flexible prompt encoder**, and **a fast mask decoder**. The image embedding obtained by the image encoder, which is a large model, can be reused in the image decoder.

The structure of the application reflects the structure of SAM. The **image encoder** works on the **server part**. It uses GPU resources to make inferences. The image encoder is only performed when a new image is uploaded.
The relatively lightweight **mask decoder** and **prompt encoder** work on the **client part**. They take the image embedding obtained by the image encoder as input.

![k8s-structure](https://github.com/MrSyee/SAM-remove-background/assets/17582508/53813067-867a-4f3d-993a-61f90024e73b)
When using the k8s cluster, set up a dashboard for monitoring log and metric with Grafana, Prometheus, and Loki.

## Run
### 1. In local with conda
Initialize conda environment.
```bash
make env
conda activate sam-remove-background
make setup
```
Download models.
```bash
make model
```

Run API Server for SAM encoder.
```bash
make server
```

Run Gradio UI.
```bash
make client
```
Browse localhost:7860.

### 2. Docker compose
Download models.
```bash
make model
```

Run services with docker-compose.
```bash
docker compose up -d
```
Browse localhost:7860.

### 3. k8s
Install the prerequisites:
- Install [Kubernetes CLI (kubectl)](https://kubernetes.io/docs/tasks/tools/).
- Install [Helm](https://helm.sh/docs/intro/install/).

Create cluster.
```bash
make cluster
```

Install helm charts.
```bash
# Set secret for ghcr auth. Required github token `secrets/token.txt`.
sh scripts/init.sh

make charts
```

Check pods.
```bash
kubectl get pods

NAME                                                     READY   STATUS    RESTARTS   AGE
minio-6649978ff8-xsssz                                   1/1     Running   0          29h
dcgm-exporter-46qph                                      1/1     Running   0          29h
prometheus-prometheus-node-exporter-znwfm                1/1     Running   0          18h
prometheus-kube-prometheus-operator-6c676cfb6b-7gfwt     1/1     Running   0          18h
alertmanager-prometheus-kube-prometheus-alertmanager-0   2/2     Running   0          18h
prometheus-kube-state-metrics-7f4f499cb5-dtkgr           1/1     Running   0          18h
prometheus-grafana-66cf6786cf-vr2cl                      3/3     Running   0          18h
prometheus-prometheus-kube-prometheus-prometheus-0       2/2     Running   0          18h
loki-0                                                   1/1     Running   0          18h
promtail-2v556                                           1/1     Running   0          18h
traefik-677c7d64f8-xq45v                                 1/1     Running   0          51m
triton-f78b5c4b7-xxslh                                   1/1     Running   0          44m
triton-prometheus-adapter-77fddcf84-tg6l4                1/1     Running   0          44m
sam-remove-background-server-5558d66455-9xzhr            1/1     Running   0          29m
sam-remove-background-client-f554f4d85-zdn85             1/1     Running   0          26m
```

Remove cluster.
```bash
make finalize
```




## Model repository
For this project, I used one of [the pre-trained SAM models](https://github.com/facebookresearch/segment-anything#model-checkpoints), the `sam_vit_h` model. The decoder was converted to onnx and the encoder was converted to torchscript for uploading to triton. Both models were uploaded to huggingface ([encoder](https://huggingface.co/khsyee/sam-vit-h-encoder-torchscript/tree/main) | [decoder](https://huggingface.co/khsyee/sam-vit-h-decoder-onnx-quantized)).
