# SAM Extract Object App
[[Video]()]

[Segment Anything Model(SAM)](https://github.com/facebookresearch/segment-anything) is the foundation model for the segmentation task by Meta.
In this repository, an application that extract an object and remove the background by utilizing SAM's interactive segmentation with click is implemented. The UI is implemented using [gradio](https://gradio.app/).
The Encoder part of SAM is served to the [triton inference server](https://github.com/triton-inference-server/server) to increase the efficiency of inference.
All parts of the app were configured for deployment on docker and k8s.


## Contents
- [Done] Implement remove background app using SAM with Gradio.
- [Done] Docker compose for the server and client.
- [Done] Convert pre-trained SAM Encoder to torchscript. ([Huggingface](https://huggingface.co/khsyee/sam-vit-h-encoder-torchscript/tree/main))
- [Done] Triton serving SAM Encoder.
- Load test on Triton server (Locust).
- Kubernetes helm charts for the server and client.
- Monitoring on K8s (Promtail + Loki & Prometheus & Grafana).

## Diagram
[]

## Run
### 1. In local with conda
```bash
make env
conda activate sam-remove-background
make setup
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
```bash
make model
docker compose up -d
```
Browse localhost:7860.


## Model repository
For this project, I used one of [the pre-trained SAM models](https://github.com/facebookresearch/segment-anything#model-checkpoints), the `sam_vit_h` model. The decoder was converted to onnx and the encoder was converted to torchscript for uploading to triton. Both models were uploaded to huggingface ([encoder](https://huggingface.co/khsyee/sam-vit-h-encoder-torchscript/tree/main) | [decoder](https://huggingface.co/khsyee/sam-vit-h-decoder-onnx-quantized)).
