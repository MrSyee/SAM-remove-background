# SAM Remove background App
[Segment Anything Model(SAM)]() is the foundation model for the segmentation task by Meta.
In this repository, an application that remove the background by utilizing SAM's one click interactive segmentation feature is implemented. The UI is implemented using [gradio]().
The Encoder part of SAM is served to the triton inference server to increase the efficiency of inference.
All parts of the app were configured for deployment on docker and k8s.

# Contents
- [] Implement remove background app using SAM with Gradio
- [] Triton serving SAM Encoder
- [] Load test on Triton server (Locust)
- [] Docker compose for the server and client.
- [] Kubernetes helm charts for the server and client.
- [] Monitoring on K8s (Promtail + Loki & Prometheus & Grafana).
