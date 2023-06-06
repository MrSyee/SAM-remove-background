# SAM Extract Object App
[Segment Anything Model(SAM)]() is the foundation model for the segmentation task by Meta.
In this repository, an application that extract an object and remove the background by utilizing SAM's interactive segmentation with click is implemented. The UI is implemented using [gradio]().
The Encoder part of SAM is served to the triton inference server to increase the efficiency of inference.
All parts of the app were configured for deployment on docker and k8s.

# Contents
- [Done] Implement remove background app using SAM with Gradio
- Convert pre-trained SAM Encoder to torchscript
- Triton serving SAM Encoder
- Load test on Triton server (Locust)
- Docker compose for the server and client.
- Kubernetes helm charts for the server and client.
- Monitoring on K8s (Promtail + Loki & Prometheus & Grafana).

# Run
1. In local
```bash
make env
conda activate sam-remove-background
make setup
```

Run API Server for SAM encoder.
```bash
make server
```

Run Gradio UI.
```bash
make client
```
