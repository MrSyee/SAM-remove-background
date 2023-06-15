import os

import cv2
import tritonclient.grpc.aio as grpcclient
import uvicorn
from fastapi import FastAPI, UploadFile

from tasks.sam import SAMImageEmbeddingResponse, SAMImageEncoder
from utils.logger import get_logger

###################
# Setups
###################
cv2.setNumThreads(1)
logger = get_logger()

app = FastAPI()

# INFERENCE_SERVER_URL (str)
inference_server_url = os.getenv("INFERENCE_SERVER_URL", "localhost:8001")
# Create an inference server client.
triton_client = grpcclient.InferenceServerClient(inference_server_url)
sam_image_encoder = SAMImageEncoder(triton_client)
logger.info("API Server is ready.")


###################
# APIs
###################
@app.get("/healthcheck")
async def healthcheck() -> bool:
    """Healthcheck."""
    return True


@app.post("/image-embedding", response_model=SAMImageEmbeddingResponse)
async def create_image_embedding(file: UploadFile) -> SAMImageEmbeddingResponse:
    """Create image embedding using SAM encoder API."""
    try:
        inference_params = {
            "model_name": "sam_torchscript_fp32",
            "model_version": "",
        }
        out = await sam_image_encoder.run(file, inference_params)

    except Exception as e:
        logger.exception(e)

    return SAMImageEmbeddingResponse(**out)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
