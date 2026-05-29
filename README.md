---
title: Image To Pixel Art 4K
emoji: 💻
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 6.3.0
app_file: app.py
pinned: true
license: gpl-3.0
short_description: Transform any image with face into retro pixel art style!
disable_embedding: false
---

# Image to Pixel Art (SDXL + LCM + Depth ControlNet)

Transform any image into beautiful retro pixel art style using AI.

## Features
- **Depth-Aware Styling:** Uses Zoe Depth ControlNet to preserve structure and depth
- **Auto-Captioning:** Automatically generates captions for your images using BLIP
- **Fast Generation:** Uses LCM Scheduler for quick results
- **Customizable:** Adjust strength, steps, and other parameters

## How It Works
1. Upload any image
2. Optionally add a custom prompt
3. Adjust settings if needed
4. Click "Generate Pixel Art"

## Technical Details
- Base Model: SDXL with custom checkpoint
- ControlNet: Zoe Depth for structure preservation
- LoRA: RetroArt style for pixel art aesthetics
- Scheduler: LCM for fast inference