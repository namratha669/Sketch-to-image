# Sketch-to-Photorealistic Image Generator

Multimodal Generative Framework for Sketch and Text to Realistic Image Conversion.
Uses Stable Diffusion (`SG161222/Realistic_Vision_V5.1_noVAE`) + ControlNet (Canny)
with a Gradio front end.

## Run on Google Colab (recommended for testing)

1. Open a new Colab notebook, set **Runtime → Change runtime type → GPU**.
2. Install deps:
   ```bash
   !pip install diffusers transformers accelerate xformers opencv-python-headless gradio huggingface_hub
   ```
3. Upload `app.py` to the Colab file browser (or `!wget`/paste it into a cell).
4. Run it:
   ```bash
   !python app.py
   ```
   Gradio will print a public `*.gradio.live` link (from `demo.launch(debug=True)` —
   add `share=True` if it doesn't appear automatically).

> Note: `Realistic_Vision_V5.1_noVAE` is not gated, so you likely don't need
> `huggingface_hub.login()`. If you swap in a gated model later, uncomment the
> login lines at the top of `app.py`.

## Run locally (VS Code) — needs an NVIDIA GPU

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Without a GPU it will run on CPU (`device = "cpu"`) but will be very slow
(minutes per image) — fine for a quick correctness check, not for real use.

## Pushing to GitHub

```bash
git init
git add app.py requirements.txt README.md
git commit -m "Initial commit: sketch-to-image generator"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Deploying "live"

Heads up: **this app can't run on Vercel.** Vercel is a serverless/static
platform (short request timeouts, no persistent GPU process), and this app
needs a long-lived Python process holding a multi-GB Stable Diffusion model
in GPU memory. Good options instead:

- **Hugging Face Spaces (Gradio SDK)** — easiest, free GPU tier available,
  built exactly for this. Push `app.py` + `requirements.txt` to a new Space.
- **Render / Railway** with a GPU-enabled instance (paid).
- Keep using the Colab public link for demos, and put a small **static
  landing page on Vercel** (project description, screenshots, link to the
  live Colab/HF Space demo, link to GitHub) if you want a polished front page.

If you want, next step can be: (1) confirm it runs in Colab, (2) set up the
GitHub repo, (3) build a simple landing page for Vercel that links out to a
Hugging Face Space running the actual model.
