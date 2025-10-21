import gradio as gr
import numpy as np
from PIL import Image
import keras
import os
import sys

model_path = "mirnet_model"

# --- Path Validation ---
if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, 'saved_model.pb')):
    print(f"--- ERROR ---")
    print(f"The path '{model_path}' is incorrect or does not contain a 'saved_model.pb' file.")
    sys.exit()
if not os.path.exists(os.path.join(model_path, 'variables')):
    print(f"--- ERROR ---")
    print(f"The 'variables' folder is missing from '{model_path}'.")
    sys.exit()

try:
    print(f"Attempting to load model from: {model_path}")
    inputs = keras.Input(shape=(None, None, 3))
    tfsm_layer = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    outputs = tfsm_layer(inputs)
    model = keras.Model(inputs, outputs)
    print("Model loaded successfully.")

except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def enhance_image(original_image):
    if model is None:
        raise gr.Error("The model is not loaded. Please check the terminal for errors.")

    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')

    # Prepare output image
    input_numpy = keras.utils.img_to_array(original_image)
    input_numpy = input_numpy.astype("float32") / 255.0
    input_batch = np.expand_dims(input_numpy, axis=0)
    output_data = model.predict(input_batch, verbose=0)
    output_numpy = list(output_data.values())[0][0]
    output_numpy = output_numpy.astype(np.float32)
    output_numpy = output_numpy[:, :, :3]
    final_numpy = input_numpy - output_numpy
    final_numpy = np.clip(final_numpy, 0.0, 1.0)
    output_image = final_numpy * 255.0

    # Create the final PIL Image
    output_image = Image.fromarray(output_image.astype(np.uint8))
    return output_image


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# MenuVision")
    with gr.Row(variant="panel"):
        with gr.Column():
            original_image_input = gr.Image(label="Upload Low-Light Image", type="pil", height=400)
            submit_button = gr.Button("Enhance Image", variant="primary")
        with gr.Column():
            enhanced_image_output = gr.Image(label="Enhanced Image", height=400)
    submit_button.click(fn=enhance_image, inputs=original_image_input, outputs=enhanced_image_output)

if __name__ == "__main__":
    if model:
        demo.launch()
    else:
        print("Gradio interface did not launch because the model failed to load.")