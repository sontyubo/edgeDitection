import gradio as gr
from PIL import Image
import cv2
import numpy as np

from utils.edge_ditect import DoG, XDoG


def cv2pil_binary(image_cv):
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_cv)
    image_pil = image_pil.convert("L")
    return image_pil


def DoG_filter(input_img, kernel_size, sigma=1.3):
    dog_img = DoG(input_img, kernel_size=kernel_size, sigma=sigma)
    dog_img_pil = cv2pil_binary(dog_img)
    return dog_img_pil


def XDoG_filter(input_img, kernel_size, epsilon, phi, sigma=1.3):
    xdog_img = XDoG(
        image=input_img, kernel_size=kernel_size, epsilon=epsilon, phi=phi, sigma=sigma
    )
    xdog_img_pil = cv2pil_binary((xdog_img * 255).astype(np.uint8))
    return xdog_img_pil


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="numpy", label="Input Image")
                kernel_size = gr.Slider(
                    minimum=1, maximum=31, value=7, step=1, label="Kernel Size"
                )
                epsilon = gr.Slider(
                    minimum=-20, maximum=30, value=10, step=1, label="Epsilon"
                )
                phi = gr.Slider(minimum=0, maximum=20, value=2, step=1, label="Phi")

        with gr.Row():
            with gr.Column():
                DoG_btn = gr.Button("Generate")
                DoG_img = gr.Image(type="pil", label="DoG Image")

            with gr.Column():
                XDoG_btn = gr.Button("Generate")
                XDoG_img = gr.Image(type="numpy", label="XDoG Image")

        DoG_btn.click(fn=DoG_filter, inputs=[input_img, kernel_size], outputs=[DoG_img])
        XDoG_btn.click(
            fn=XDoG_filter,
            inputs=[input_img, kernel_size, epsilon, phi],
            outputs=[XDoG_img],
        )

    demo.launch(debug=True)
