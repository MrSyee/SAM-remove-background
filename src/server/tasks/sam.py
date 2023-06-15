import base64
import os
import urllib
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
import tritonclient.grpc.aio as grpcclient
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
    def __init__(self, triton_client: grpcclient.InferenceServerClient) -> None:
        logger.info("Initialize SAMImageEncoder.")
        self.triton_client = triton_client

    @torch.no_grad()
    async def run(
        self, file: UploadFile, inference_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        logger.info("Run SAMImageEncoder.")
        image = await file.read()

        # Preprocess
        input_image = self.preprocess(image)

        # Prepare for inference.
        triton_inputs = [grpcclient.InferInput("INPUT__0", input_image.shape, "FP32")]
        triton_inputs[0].set_data_from_numpy(input_image)
        triton_outputs = [grpcclient.InferRequestedOutput("OUTPUT__0")]

        # Run the inference.
        result = await self.triton_client.infer(
            inputs=triton_inputs,
            outputs=triton_outputs,
            **inference_params,
        )
        # image embedding: numpy.ndarray [B, 256, 64, 64]
        image_embedding = result.as_numpy("OUTPUT__0")

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
        if image.shape != (1024, 1024, 3):
            origin_shape = image.shape[:2]
            height, width = self.get_preprocess_shape(*origin_shape)
            image = cv2.resize(image, dsize=(width, height))
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

        return preprocessed

    def postprocess(self, image_embedding: torch.Tensor) -> Dict[str, Any]:
        """Postprocess the inference results for exporting as API response."""
        image_embedding_shape = image_embedding.shape
        image_embedding = base64.b64encode(image_embedding.tobytes()).decode("utf8")
        return {
            "image_embedding": image_embedding,
            "image_embedding_shape": image_embedding_shape,
        }

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int = 1024
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
