import base64
import logging
import os

import cv2
import gradio as gr
import httpx
import numpy as np

###################
# Setups
###################
cv2.setNumThreads(1)
logger = logging.getLogger()

EXAMPLES_DIR = "assets/examples"
API_SERVER_URL = os.environ.get("API_SERVER_URL", "http://localhost:8888")


###################
# Events
###################
async def create_image_embedding(image: np.ndarray):
    """"""
    print(image.shape)
    # Encode image to byte
    image_bytes = cv2.imencode(".jpg", image)[1].tobytes()

    files = {"file": image_bytes}

    # Inference model using API
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{API_SERVER_URL}/image-embedding", files=files
            )
            print("status_code", response.status_code)
        if response.status_code == 200:
            resp = response.json()

    except Exception as e:
        logger.exception(e)

    # Decode
    image_embedding = base64.b64decode(resp["image_embedding"])
    image_embedding = np.frombuffer(image_embedding, dtype=np.float32)

    # Save embedding by numpy format
    os.makedirs("assets", exist_ok=True)
    filepath = "assets/embedding.npy"
    np.save(filepath, image_embedding)

    return filepath


def extract_object(image: np.ndarray, point_h: int, point_w: int, point_label: int):
    pass


def get_coords(evt: gr.SelectData):
    return evt.index[0], evt.index[1]


###################
# UI
###################
with gr.Blocks() as app:
    print("[INFO] Gradio app ready")
    gr.Markdown("# Interactive Extracting Object from Image")

    gr.Markdown("## Image")
    with gr.Row():
        coord_h = gr.Number(label="Mouse coords h")
        coord_w = gr.Number(label="Mouse coords w")

    with gr.Row():
        input_img = gr.Image(label="Input image").style(height=600)
        output_img = gr.Image(label="Output image").style(height=600)

    with gr.Row():
        embed_btn = gr.Button(value="Create image embedding")

    gr.Markdown(
        """
        ## Image Embedding file
        To inference model you should get image embedding that is generated
            by SAM encoder.

        You can setup image embedding in two ways.
        1. after uploading the image, press the Create image embedding button.
        2. manually upload an image embedding file for the uploaded image.
            In this case, the image embedding file must be in npy format.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_embedding_file = gr.File(label="image embedding file", type="binary")

    # Create image embedding when upload input image
    embed_btn.click(
        create_image_embedding,
        inputs=input_img,
        outputs=image_embedding_file,
    )

    # Extract object and remove background when click object
    input_img.select(get_coords, None, [coord_h, coord_w])

    gr.Markdown("## Image Examples")
    gr.Examples(
        examples=[[os.path.join(EXAMPLES_DIR, "mannequin.jpeg")]],
        inputs=[input_img],
        fn=create_image_embedding,
        run_on_click=True,
    )

if __name__ == "__main__":
    app.launch()
