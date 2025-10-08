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


## 🔍 Under the hood


### `PersonalizedGRPOTrainer` (extends `trl.GRPOTrainer`)
* Accepts an `image_processor` and an `images_root` folder.  
* Overrides **`data_collator`** to  
  1. Load images with **Pillow**.  
  2. Batch‑encode them via the Hugging Face `AutoProcessor`.  
  3. Return a dict containing  
     * `pixel_values` – tensor *(C × H × W)*  
     * `prompt` – instruction string  
     * `solution` – ground‑truth bbox or coordinates  
     * `scales` – original image size  

### `Qwen2_5_VLForConditionalGenerationWithLogits` (wrapper)
Tiny subclass that forwards all arguments to the real **Qwen 2.5‑VL** model while gracefully ignoring the extra `logits_to_keep` parameter expected by GRPO.

### Reward functions (`rewards.py`)
Currently only **`accuracy_reward_coord`**, which returns **1** if the (x, y) coordinate predicted by the model falls inside the ground‑truth bounding‑box and **0** otherwise.  
Feel free to add IoU‑ or distance‑based rewards here.

### Training script (`main.py`)
Provides a concrete example wiring everything together.  
Customise the constants at the top, or replace them with **argparse** flags for production use.

---

## ⚙️ Configuration tips

| Hyper‑parameter                   | Where to set   | Notes                                                     |
|----------------------------------|----------------|-----------------------------------------------------------|
| `per_device_train_batch_size`    | `GRPOConfig`   | Limited by GPU memory – images are heavy!                |
| `num_generations`                | `GRPOConfig`   | How many action samples to draw per prompt.              |
| `reward_funcs`                   | trainer init   | List of callables returning a reward ∈ {0, 1}.           |
| `bf16` / `fp16`                  | `GRPOConfig`   | Use `bf16` on A100/H100 for speed and memory efficiency. |

---


