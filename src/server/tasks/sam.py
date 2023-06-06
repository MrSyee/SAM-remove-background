import base64
import os
import urllib
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
from fastapi import UploadFile
from pydantic import BaseModel, Field
from segment_anything import sam_model_registry

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
    def __init__(
            self, checkpoint_path, checkpoint_name, checkpoint_url, model_type
        ) -> None:
        logger.info("Initialize SAMImageEncoder.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint = os.path.join(checkpoint_path, checkpoint_name)
        if not os.path.exists(checkpoint):
            urllib.request.urlretrieve(checkpoint_url, checkpoint)
        self.model = sam_model_registry[model_type](checkpoint=checkpoint).to(
            self.device
        )
        logger.info("Complete to initialize SAMImageEncoder.")

    @torch.no_grad()
    async def run(self, file: UploadFile) -> Dict[str, Any]:
        logger.info("Run SAMImageEncoder.")

        image = await file.read()

        # Preprocess
        input_image = self.preprocess(image)

        # Inference
        # image embedding: torch.Tensor, [B, 256, 64, 64]
        image_embedding = self.model.image_encoder(input_image)
        # numpy.ndarray [B, 256, 64, 64]
        image_embedding = image_embedding.cpu().detach().numpy()

        # Postprocess
        outputs = self.postprocess(image_embedding)

        return outputs

    def preprocess(self, image_byte, target_size=1024) -> torch.Tensor:
        """
        Preprocess image to input to encoder.

        Return:
            preprocessed: If longest size of target image is 1024,
                        shape of tensor is [B, 3, 1024, 1024].
                        And dtype of tensor is float32.
        """
        # Convert the bytes to numpy array
        image = np.frombuffer(image_byte, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)[:, :, ::-1]  # RGB

        # Get image shape and convert type
        height, width = image.shape[:2]
        image_fp = image.astype(np.float32)

        # Normalize
        image_fp -= np.array([123.675, 116.28, 103.53], dtype=np.float32)  # mean
        image_fp /= np.array([58.395, 57.12, 57.375], dtype=np.float32)  # std

        # Padding
        preprocessed = np.zeros((target_size, target_size, 3), dtype=np.float32)
        preprocessed[:height, :width, :] = image_fp

        # Convert torch tensor
        preprocessed = np.moveaxis(preprocessed, -1, 0)[None, :, :, :]
        preprocessed = torch.tensor(preprocessed).to(self.device)

        return preprocessed

    def postprocess(self, image_embedding: torch.Tensor) -> Dict[str, Any]:
        """Postprocess the inference results for exporting as API response."""
        image_embedding_shape = image_embedding.shape
        image_embedding = base64.b64encode(image_embedding.tobytes()).decode("utf8")
        return {
            "image_embedding": image_embedding,
            "image_embedding_shape": image_embedding_shape,
        }
