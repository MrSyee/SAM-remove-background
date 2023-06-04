import os
import urllib
from typing import Any, Dict, Tuple

import numpy as np
import torch
from fastapi import UploadFile
from pydantic import BaseModel, Field
from segment_anything import SamPredictor, sam_model_registry

from utils.logger import get_logger

EmbeddingShape = Tuple[int, int, int, int]

logger = get_logger()

# Model
class SAMImageEmbeddingResponse(BaseModel):
    """SAM Image embedding Response model."""

    image_embedding: str = Field(..., description="Image Embedding")
    image_embedding_shape: EmbeddingShape = Field(..., example=[1, 256, 64, 64])


# Controller
class SAMImageEncoder:
    def __init__(self, checkpoint_path, checkpoint_name, checkpoint_url, model_type):
        logger.info("Initialize SAMImageEncoder.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint = os.path.join(checkpoint_path, checkpoint_name)
        if not os.path.exists(checkpoint):
            urllib.request.urlretrieve(checkpoint_url, checkpoint)
        sam = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
        self.predictor = SamPredictor(sam)

    @torch.no_grad()
    async def run(self, file: UploadFile) -> Dict[str, Any]:
        logger.info("Run SAMImageEncoder.")

        image = await file.read()

        # Preprocess

        # Inference
        # self.predictor.set_image()

        # Postprocess

        return {"image_embedding": "", "image_embedding_shape": [1, 256, 64, 64]}
