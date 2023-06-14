import base64
import logging
import multiprocessing
import os
from typing import Tuple

import cv2
import gradio as gr
import httpx
import numpy as np
import onnxruntime as ort

from utils import get_preprocess_shape

###################
# Setups
###################
cv2.setNumThreads(1)
logger = logging.getLogger()

EXAMPLES_DIR = os.environ.get("EXAMPLE_DIR", "assets/examples")
API_SERVER_URL = os.environ.get("API_SERVER_URL", "http://localhost:8888")
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH", "checkpoint/sam_onnx_quantized.onnx"
)

IMAGE_SIZE_FOR_EMBEDDING = 1024
POINTS_LABELS = np.array([[1, -1]], dtype=np.float32)
MASK_INPUT = np.zeros((1, 1, 256, 256), dtype=np.float32)
HAS_MASK_INPUT = np.zeros(1, dtype=np.float32)

# Set decoder with ONNXRuntime
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = min(8, multiprocessing.cpu_count() // 2)
ort_sess = ort.InferenceSession(CHECKPOINT_PATH, sess_options)

###################
# Events
###################
async def get_image_embedding(image: np.ndarray) -> str:
    """Get image embedding using API."""
    print("[INFO] Get image embedding.")
    # Resize the image while maintaining the aspect ratio
    origin_shape = image.shape[:2]
    height, width = get_preprocess_shape(*origin_shape, IMAGE_SIZE_FOR_EMBEDDING)
    image = cv2.resize(image, dsize=(width, height))

    # Encode image to byte
    image_bytes = cv2.imencode(".jpg", image)[1].tobytes()

    files = {"file": image_bytes}

    # Inference model using API
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{API_SERVER_URL}/image-embedding", files=files
            )
        if response.status_code == 200:
            resp = response.json()
        else:
            print("[Error] status_code is not 200: ", response.status_code)

    except Exception as e:
        logger.exception(e)

    # Decode
    image_embedding = base64.b64decode(resp["image_embedding"])
    image_embedding = np.frombuffer(image_embedding, dtype=np.float32)

    # Save embedding by numpy format
    os.makedirs("outputs", exist_ok=True)
    filepath = "outputs/embedding.npy"
    np.save(filepath, image_embedding)

    return filepath


def extract_object(
    image: np.ndarray,
    embedding_file: str,
    point_w: int,
    point_h: int,
) -> np.ndarray:
    """Extract object and remove background to inference model."""
    # Load embedding
    image_embedding = np.load(embedding_file.name)
    image_embedding = image_embedding.reshape(1, 256, 64, 64)

    # Resize the image to prevent too large image
    image_shape = image.shape[:-1]

    # Corrects the coordinates for the resize
    old_h, old_w = image_shape
    new_h, new_w = get_preprocess_shape(old_h, old_w, IMAGE_SIZE_FOR_EMBEDDING)
    point_w *= new_w / old_w
    point_h *= new_h / old_h

    # Get mask to inference model
    points_coords = np.array([[(point_w, point_h), (0, 0)]], dtype=np.float32)
    orig_im_size = np.array(image_shape, dtype=np.float32)
    masks, _, _ = ort_sess.run(
        None,
        {
            "image_embeddings": image_embedding,
            "point_coords": points_coords,
            "point_labels": POINTS_LABELS,
            "mask_input": MASK_INPUT,
            "has_mask_input": HAS_MASK_INPUT,
            "orig_im_size": orig_im_size,
        },
    )
    mask = masks[0, 0, :, :]

    # Postprocess mask
    mask = (mask > 0).astype(np.uint8)

    # Remove background
    result_image = cv2.bitwise_and(image, image, mask=mask)

    # Convert to rgba channel
    bgr_channel = result_image[..., :3]  # BGR 채널 분리
    alpha_channel = np.where(bgr_channel[..., 0] == 0, 0, 255).astype(
        np.uint8
    )  # 투명도 채널 생성
    result_image = np.dstack((bgr_channel, alpha_channel))  # BGRA 이미지 생성

    return result_image


def extract_object_by_event(
    image: np.ndarray, embedding_file: str, evt: gr.SelectData
) -> np.ndarray:
    """Extract object by click event."""
    click_h, click_w = evt.index
    return extract_object(image, embedding_file, click_h, click_w)


def get_coords(evt: gr.SelectData) -> Tuple[int, int]:
    """Get coords by click event."""
    return evt.index[0], evt.index[1]


###################
# UI
###################
with gr.Blocks() as app:
    print("[INFO] Gradio app ready")
    gr.Markdown("# Interactive Extracting Object from Image")

    gr.Markdown("## Image")
    with gr.Row():
        coord_x = gr.Number(label="Mouse coords x")
        coord_y = gr.Number(label="Mouse coords y")

    with gr.Row():
        input_img = gr.Image(label="Input image").style(height=600)
        output_img = gr.Image(label="Output image").style(height=600)

    with gr.Row():
        embed_btn = gr.Button(value="Get image embedding")

    gr.Markdown(
        """
        ## Image Embedding file
        To inference model you should get image embedding that is generated
            by SAM encoder.

        You can setup image embedding in two ways.
        1. after uploading the image, press the 'Get image embedding' button.
        2. manually upload an image embedding file for the uploaded image.
            In this case, the image embedding file must be in npy format.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_embedding_file = gr.File(label="image embedding file", type="file")

    # Create image embedding when upload input image
    embed_btn.click(
        get_image_embedding,
        inputs=input_img,
        outputs=image_embedding_file,
    )

    # Extract object and remove background when click object
    input_img.select(
        extract_object_by_event, [input_img, image_embedding_file], output_img
    )
    input_img.select(get_coords, None, [coord_x, coord_y])

    gr.Markdown("## Image Examples")
    gr.Examples(
        examples=[
            [
                os.path.join(EXAMPLES_DIR, "mannequin.jpeg"),
                os.path.join(EXAMPLES_DIR, "embedding.npy"),
                1720,
                230,
            ]
        ],
        inputs=[input_img, image_embedding_file, coord_x, coord_y],
        outputs=output_img,
        fn=extract_object,
        run_on_click=True,
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0")
