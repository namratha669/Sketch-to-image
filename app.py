"""
Sketch-to-Photorealistic Image Generator
Multimodal Generative Framework for Sketch and Text to Realistic Image Conversion
(Extracted / cleaned up from the mini project report)

Pipeline: Stable Diffusion + ControlNet (Canny edges) + Gradio UI
"""

import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from PIL import Image
import numpy as np
import cv2
import gradio as gr

# -----------------------------------------------------------------------
# If running on Google Colab and the base model is gated on Hugging Face,
# uncomment these two lines and log in with your HF token first:
# from huggingface_hub import login
# login()
# -----------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

base_model_id = "SG161222/Realistic_Vision_V5.1_noVAE"

# Use ONLY Canny ControlNet (simpler, more reliable)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

# Use better scheduler for quality
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Enable memory optimizations
if device == "cuda":
    pipe.enable_xformers_memory_efficient_attention()

print("Model loaded successfully")


# -----------------------------------------------------------------------
# Preprocessing helpers
# -----------------------------------------------------------------------

def preprocess_for_faces(image):
    """Multi-stage edge detection to capture ALL facial features"""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    img_array = np.array(image)
    if len(img_array.shape) == 2:
        gray = img_array
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Apply slight blur to reduce noise but preserve features
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Multi-level edge detection
    very_subtle = cv2.Canny(gray, 20, 60)   # fine details
    medium = cv2.Canny(gray, 50, 120)       # main features
    strong = cv2.Canny(gray, 80, 180)       # outline

    # Combine all levels - this captures EVERYTHING
    combined = cv2.bitwise_or(very_subtle, medium)
    combined = cv2.bitwise_or(combined, strong)

    # Slightly dilate to strengthen edges
    kernel = np.ones((2, 2), np.uint8)
    combined = cv2.dilate(combined, kernel, iterations=1)

    edges_rgb = cv2.cvtColor(combined, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb), Image.fromarray(combined)


def preprocess_general(image):
    """Standard preprocessing for non-face content"""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    img_array = np.array(image)
    if len(img_array.shape) == 2:
        gray = img_array
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb), Image.fromarray(edges)


def build_prompt(description, angle, content_type):
    """Build optimized prompts based on content type"""
    if content_type == "face":
        prompt = (
            f"professional portrait photograph, {description}, {angle} view, "
            "RAW photo, face photo, highly detailed facial features, "
            "EXACT facial structure from sketch, precise facial proportions, "
            "maintain face shape, accurate face geometry, "
            "realistic skin texture, natural skin pores, "
            "symmetric face, professional headshot, studio lighting, "
            "8k uhd, dslr, high quality, sharp focus, Fujifilm XT3"
        )
        negative_prompt = (
            "different face, wrong face shape, altered proportions, "
            "drawing, painting, sketch, illustration, anime, cartoon, "
            "deformed face, distorted face, asymmetric face, ugly face, "
            "deformed iris, deformed pupils, cross-eyed, bad eyes, "
            "bad anatomy, wrong anatomy, mutated, disfigured, "
            "blurry, soft focus, low quality, watermark, text"
        )
    elif content_type == "object":
        prompt = (
            f"professional product photography, {description}, {angle} view, "
            "accurate geometry, correct proportions, precise structure, "
            "maintain exact shape from sketch, maintain exact arrangement, "
            "clean lines, sharp edges, photorealistic, highly detailed, "
            "studio lighting, 8k uhd, product shot, professional photography, sharp focus"
        )
        negative_prompt = (
            "wrong colors, incorrect colors, mixed colors, random colors, "
            "distorted, warped, bent, twisted, deformed, incorrect proportions, "
            "wrong shape, blurry, painting, drawing, sketch, illustration, "
            "low quality, bad geometry, asymmetric, watermark, text"
        )
    else:
        prompt = (
            f"RAW photo of {description}, {angle} view, "
            "photorealistic, highly detailed, natural lighting, sharp focus, 8k uhd, "
            "professional photography, realistic textures, high quality"
        )
        negative_prompt = (
            "blurry, painting, drawing, cartoon, anime, illustration, "
            "low quality, distorted, deformed, ugly, bad anatomy, watermark, text"
        )

    return prompt, negative_prompt


def generate_image(sketch, description, angle, edge_strength, content_type, seed):
    """Generate image with appropriate preprocessing"""

    if seed == -1:
        seed = np.random.randint(0, 2147483647)
    generator = torch.Generator(device=device).manual_seed(int(seed))

    if content_type == "face":
        control_image, edge_viz = preprocess_for_faces(sketch)
        num_steps = 50
        guidance = 9.5
    elif content_type == "object":
        control_image, edge_viz = preprocess_for_faces(sketch)  # detailed edges for objects too
        num_steps = 40
        guidance = 8.5
        edge_strength = min(edge_strength * 1.2, 2.0)
    else:
        control_image, edge_viz = preprocess_general(sketch)
        num_steps = 30
        guidance = 7.5

    prompt, negative_prompt = build_prompt(description, angle, content_type)

    if device == "cuda":
        with torch.autocast("cuda"):
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                controlnet_conditioning_scale=edge_strength,
                generator=generator,
            ).images[0]
    else:
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                controlnet_conditioning_scale=edge_strength,
                generator=generator,
            ).images[0]

    return result, edge_viz, seed


def infer(sketch, description, angle, edge_strength, content_type, seed):
    if sketch is None:
        return None, None, "Upload a sketch first!"

    generated_img, edge_img, used_seed = generate_image(
        sketch, description, angle, edge_strength, content_type, seed
    )
    return generated_img, edge_img, f"Seed used: {used_seed}"


# -----------------------------------------------------------------------
# Gradio UI
# -----------------------------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown("# Sketch-to-Photorealistic Image (Face Optimized)")
    gr.Markdown(
        """
        For BEST face similarity:
        - Select 'face' mode
        - Use edge strength 1.2-1.4 (higher = more accurate to sketch)
        - Be very specific: "young man with short brown hair, blue eyes, oval face"
        - Check "Detected Edges" - if your sketch features aren't showing, redraw with darker lines

        For animals/general content:
        - Select 'general' mode
        - Use edge strength 0.7-0.9 (lower = more natural looking)
        - Simple descriptions work best: "golden retriever puppy sitting"
        """
    )

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Upload Your Sketch", type="pil")

            content_type = gr.Radio(
                ["general", "face", "object"],
                value="general",
                label="Content Type",
                info="face=portraits | object=things like books, furniture, etc",
            )

            description = gr.Textbox(
                label="Detailed Description",
                placeholder="For faces: 'young woman with long curly black hair, brown eyes, round face, smiling'",
                value="a white dog in garden",
                lines=2,
            )

            with gr.Row():
                angle = gr.Radio(
                    ["front", "side", "three-quarter", "closeup"],
                    value="front",
                    label="Camera Angle",
                )
                edge_strength = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=0.75,
                    step=0.05,
                    label="Edge Strength (GENERAL: 0.7-0.9 | FACES: 1.2-1.5 | OBJECTS: 1.0-1.3)",
                )

            seed = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)

            generate_btn = gr.Button("Generate Photo", variant="primary", size="lg")

        with gr.Column():
            output_img = gr.Image(label="Generated Photorealistic Image")
            edge_img = gr.Image(label="Detected Edges (Check if features captured)")
            seed_output = gr.Textbox(label="Seed Info", interactive=False)

        generate_btn.click(
            fn=infer,
            inputs=[input_img, description, angle, edge_strength, content_type, seed],
            outputs=[output_img, edge_img, seed_output],
        )

    gr.Markdown(
        """
        ### Troubleshooting:
        | Problem | Solution |
        |---|---|
        | Face doesn't match sketch | Increase edge strength to 1.4-1.8 |
        | Face looks weird/distorted | Lower edge strength to 1.0-1.2 |
        | Missing facial features | Make sketch lines darker and clearer |
        | Wrong gender/age | Be MORE specific in description |
        | Objects look warped | Select 'object' mode, use edge strength 1.0-1.3 |
        | Objects losing shape | Make sketch lines clearer and more defined |

        ### Example Descriptions:
        **For Faces:**
        - "middle-aged man with short grey hair, glasses, square jaw, serious expression"
        - "young woman with long wavy blonde hair, green eyes, oval face, natural smile"

        **For Objects:**
        - "stack of old leather-bound books, top book is burgundy red, middle book is dark brown, bottom book is deep green"
        - "three hardcover books stacked together with golden decorative spines"
        - "wooden chair with curved backrest and carved details"
        - "vintage camera with metal body and leather grip"

        For multiple colored objects: Be VERY specific about which object has which color, and use
        descriptive color names like "burgundy red" instead of just "red"
        """
    )

if __name__ == "__main__":
    demo.launch(debug=True)
