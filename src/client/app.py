import os

import cv2
import gradio as gr
import httpx
import numpy as np

EXAMPLES_DIR = "assets/examples"
API_SERVER_URL = os.environ.get("API_SERVER_URL", "http://localhost:8888")


async def create_image_embedding(image: np.ndarray):
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
        print(e)

    # Decode

    # Remove background

    return image


def extract_object(image: np.ndarray, point_h: int, point_w: int, point_label: int):
    pass


def get_coords(evt: gr.SelectData):
    return evt.index[0], evt.index[1]


with gr.Blocks() as app:
    print("[INFO] Gradio app ready")
    gr.Markdown("# Interactive Extracting Object from Image")
    with gr.Row():
        coord_h = gr.Number(label="Mouse coords h")
        coord_w = gr.Number(label="Mouse coords w")
        click_label = gr.Number(label="label")

    with gr.Row():
        input_img = gr.Image(label="Input image").style(height=600)
        output_img = gr.Image(label="Output image").style(height=600)

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
