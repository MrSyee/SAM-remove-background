import os

import cv2
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

configs = dict(
    checkpoint_path=os.path.join("checkpoint"),
    checkpoint_name="sam_vit_h_4b8939.pth",
    checkpoint_url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    model_type="default",
)

sam_image_encoder = SAMImageEncoder(**configs)


###################
# APIs
###################
@app.get("/healthcheck")
async def healthcheck() -> bool:
    """Healthcheck."""
    logger.info("Health Check.")
    return True


@app.post("/image-embedding", response_model=SAMImageEmbeddingResponse)
async def create_image_embedding(file: UploadFile) -> SAMImageEmbeddingResponse:
    """Create image embedding using SAM encoder API."""
    logger.info("create_image_embedding.")
    try:
        out = await sam_image_encoder.run(file)

    except Exception as e:
        logger.exception(e)

    return SAMImageEmbeddingResponse(**out)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
