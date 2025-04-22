# 🖼️ GRPO Images Trainer

Fine‑tune **Qwen 2.5‑VL** (or any Vision‑Language model with the same API) on image *grounding* tasks using **GRPO (Generic Reward Prediction Optimization)** in just a few lines of code.

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" />
</p>

---

## ✨ Why use this repo?

* **Plug‑and‑play trainer** – drop in your own JSON dataset of prompts + bounding‑boxes and start training.  
* **Image‑aware data collator** – automatically loads, preprocesses and batches images.  
* **Reward‑based optimisation** – leverages the `trl` library’s GRPO algorithm for RL‑style fine‑tuning.  
* **Minimal codebase** – only three Python files, easy to read and customise.  

---

## 🗄 Repository layout

```text
.
├── GRPOImagesTrainer.py   # custom trainer + model wrapper
├── rewards.py             # reward functions
└── main.py                # training entry‑point

