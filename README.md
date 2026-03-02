# CODECRAFT_GA_05 — Neural Style Transfer (Fast Arbitrary Style Transfer)

> Apply the artistic **style** of one image (e.g., a famous painting) to the **content** of another image using a pretrained neural style transfer network, implemented in **TensorFlow** with **TensorFlow Hub** in Google Colab.

---

## Overview

This project demonstrates **Neural Style Transfer (NST)**: generating a new image that preserves the **structure/content** of a photo while adopting the **textures, colors, and brush-like patterns** of a style image.

Instead of running the classic (slow) optimization-based NST, this notebook uses a **fast arbitrary style transfer model** from **TensorFlow Hub** (`magenta/arbitrary-image-stylization-v1-256`). It performs style transfer in a single forward pass—no training required.

---

## How It Works

```
Content Image  ───────────┐
                          ├──► Style Transfer Network (TF Hub) ───► Stylized Output
Style (Painting) Image ───┘
```

| Component         | Role                                                        |
| ----------------- | ----------------------------------------------------------- |
| **Content image** | Provides layout, shapes, objects, and overall structure     |
| **Style image**   | Provides colors, textures, brush strokes, artistic patterns |
| **TF Hub model**  | Combines both into a stylized output image                  |

---

## Project Structure

```
CODECRAFT_GA_05/
│
├── GA_05.ipynb         # Main Colab notebook
├── content.jpg         # Content Image
├── style.jpg           # Style image
├── stylized_output.jpg # Output image
└── README.md           # This file
```

---

## Getting Started

### Run in Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Open `GA_05.ipynb` in Google Colab
2. (Optional) Enable GPU: `Runtime → Change runtime type → T4 GPU`

   * Works on CPU too, but GPU is faster
3. Run all cells (`Runtime → Run all`)
4. The notebook auto-downloads **sample content + style images** — no dataset needed

---

## Notebook Cells

| Cell       | Description                                                                  |
| ---------- | ---------------------------------------------------------------------------- |
| **Cell 1** | Install dependencies + imports                                               |
| **Cell 2** | Helper functions: load image, resize, display, save output                   |
| **Cell 3** | Load content & style images (default download + optional upload auto-detect) |
| **Cell 4** | Load TF Hub model and generate stylized image                                |
| **Cell 5** | Save output (`stylized_output.png`) and download it from Colab               |

---

## Images Used (Default Demo)

The default run uses TensorFlow-hosted sample images:

* **Content:** Yellow Labrador photo
* **Style:** Kandinsky “Composition 7”

You can replace either with your own images.

---

## Preprocessing

Before passing images to the model:

* Converted to **RGB**
* Resized to a maximum dimension of `512` while keeping aspect ratio
* Normalized to `[0, 1]`
* Expanded to include batch dimension: `(1, H, W, 3)`

This keeps runtime efficient and avoids memory spikes.

---

## Output

The notebook generates:

* `stylized_output.png`

The output resolution matches the processed content image size (after resizing).

---

## Customization

### Use your own images (no renaming needed)

Cell 3 includes an optional upload mode that auto-detects filenames:

* First selected file = **content**
* Second selected file = **style**

Just uncomment:

```python
from google.colab import files
uploaded = files.upload()
uploaded_names = list(uploaded.keys())
CONTENT_PATH = uploaded_names[0]
STYLE_PATH   = uploaded_names[1]
```

### Improve speed / quality tradeoff

Change the `max_dim` in `load_image()`:

* Lower (e.g., `384`) → faster
* Higher (e.g., `768`) → more detail (may be slower)

---

## Key Concepts

* **Neural Style Transfer (NST)** — merges content structure with artistic appearance
* **Content vs Style** — content preserves geometry; style transfers textures/colors
* **Fast NST** — uses a pretrained model for real-time inference (no optimization loop)
* **Arbitrary style transfer** — supports *any* painting/style image at inference time

---

## References

* [Towards Data Science — How do Neural Style Transfers work?](https://towardsdatascience.com/how-do-neural-style-transfers-work-b76de101eb3)
* [GeeksforGeeks — Neural Style Transfer with TensorFlow](https://www.geeksforgeeks.org/deep-learning/neural-style-transfer-with-tensorflow/)
* [TF Hub Model — Arbitrary Image Stylization](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)

---
