import gradio as gr
from PIL import Image
import cv2

from utils.edge_ditect import DoG


def cv2pil_binary(image_cv):
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_cv)
    image_pil = image_pil.convert("L")
    return image_pil


def DoG_filter(input_img, kernel_size, sigma=1.3):
    dog_img = DoG(input_img, size=kernel_size, sigma=sigma)
    dog_img_pil = cv2pil_binary(dog_img)
    return dog_img_pil


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(type="numpy", label="Input Image")
                kernel_size = gr.Slider(
                    minimum=1, maximum=31, value=7, step=1, label="Kernel Size"
                )
                generate_btn = gr.Button("Generate")

            with gr.Column():
                output_img = gr.Image(type="pil", label="Output Image")

        generate_btn.click(
            fn=DoG_filter, inputs=[input_img, kernel_size], outputs=[output_img]
        )

    demo.launch(debug=True)
