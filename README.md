# ✏️ → 📸 Sketch-to-Photorealistic Image Generator

Turn a rough doodle + a sentence of description into a photorealistic image.
No fancy drawing skills needed — just stick figures and vibes.

**Stack:** Stable Diffusion (`Realistic_Vision_V5.1_noVAE`) + ControlNet (Canny edges) + Gradio

The core trick: run your sketch through multi-level Canny edge detection so
the model gets a clean structural skeleton to follow, then let Stable
Diffusion + ControlNet fill in the realism while your text prompt drives the
style/details. Face mode, object mode, and general mode each get their own
tuned prompt templates and guidance settings, because a face and a coffee
mug do *not* want the same edge strength.

---

## 🚀 Quickstart (Google Colab — do this first)

1. New Colab notebook → `Runtime → Change runtime type → GPU` (T4 is fine).
2. Install the deps:
   ```bash
   !pip install diffusers transformers accelerate xformers opencv-python-headless gradio huggingface_hub
   ```
3. Drop `app.py` into the Colab file browser.
4. Run it:
   ```bash
   !python app.py
   ```
   Gradio spits out a public `*.gradio.live` link — click it, upload a sketch, go wild.


## 💻 Run it locally instead

Need an NVIDIA GPU or this will crawl.

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

CPU-only works too, it'll just generate an image approximately once per
coffee break instead of once every 7 seconds.

## 🔮 What's next

Roadmap ideas if you want to keep building on this:
- Real-time canvas drawing instead of file upload
- Multi-ControlNet (depth, pose, scribble) for finer control
- 4K upscaling with Real-ESRGAN
- Style presets (anime, oil painting, concept art)
- Ditch Gradio for a custom React front end once the pipeline is solid

Built as a mini project for AI & ML @ RNSIT. Sketch responsibly.
