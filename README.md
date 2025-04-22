# GRPO Images Trainer

A simple, customizable trainer for fine‑tuning the Qwen2.5 Visual‑Linguistic model on image grounding tasks using GRPO (Generic Reward Prediction Optimization).

---

## 📦 Installation

Make sure you have Python 3.8+ and `pip` installed. Then:

```bash
pip install torch transformers datasets trl Pillow
🗂️ Repository Structure
pgsql
Copier
Modifier
.
├── GRPOImagesTrainer.py    # Custom trainer and model wrapper
└── examples/               # (optional) scripts for data preparation & training
📝 GRPOImagesTrainer.py Overview
This file defines two key classes:

PersonalizedGRPOTrainer
Extends trl.GRPOTrainer to support image‑based inputs:

Constructor

python
Copier
Modifier
def __init__(..., image_processor, images_root, **kwargs):
    assert image_processor is not None
    assert images_root is not None
    self.image_processor = image_processor
    self.images_root = images_root
    super().__init__(**kwargs)
    self.data_collator = self._image_data_collator
image_processor: a Hugging Face AutoProcessor (handles resizing, normalization, etc.)

images_root: root directory containing all image files referenced by your dataset.

_image_data_collator
Custom collator that, for each batch:

Loads images from images_root given filenames in features[i]['img_filename'].

Converts and batches them via image_processor → pixel_values tensor.

Extracts prompts (instruction) and ground‑truth bounding boxes (bbox).

Returns a list of dicts with keys:

pixel_values: (C, H, W) tensor

prompt: the instruction string

solution: the bounding‑box labels (list of ints)

scales: original image (width, height) tuple

Qwen2_5_VLForConditionalGenerationWithLogits
Thin subclass of the Hugging Face Qwen2_5_VLForConditionalGeneration model:

python
Copier
Modifier
class Qwen2_5_VLForConditionalGenerationWithLogits(Qwen2_5_VLForConditionalGeneration):
    def forward(self, *args, logits_to_keep: int | None = None, **kwargs):
        # ignore logits_to_keep, forward everything else
        return super().forward(*args, **kwargs)
Allows passing an extra logits_to_keep argument (used by TRL) without error.

🚀 Quick Start
python
Copier
Modifier
from transformers import AutoProcessor, AutoTokenizer
from datasets import load_dataset
from trl import GRPOConfig
from GRPOImagesTrainer import PersonalizedGRPOTrainer, Qwen2_5_VLForConditionalGenerationWithLogits

# 1. Load your image-grounding dataset
#    Expect each example to have:
#      - 'img_filename': path relative to images_root
#      - 'instruction': text prompt
#      - 'bbox': list of [x0, y0, x1, y1] ints
dataset = load_dataset("json", data_files="data/train.json")["train"]

# 2. Prepare model & processor
model_name = "Qwen/Qwen-2.5-VL"
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2_5_VLForConditionalGenerationWithLogits.from_pretrained(model_name)

# 3. Configure GRPO
config = GRPOConfig(
    learning_rate=5e-5,
    batch_size=4,
    # … other GRPO hyperparameters …
)

# 4. Instantiate your trainer
trainer = PersonalizedGRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,           # or provide a validation split
    config=config,
    image_processor=processor,
    images_root="/path/to/images"
)

# 5. Start training
trainer.train()
🤝 Contributing
Fork the repo

Create a feature branch (git checkout -b feature/YourFeature)

Commit your changes (git commit -m "Add cool feature")

Push to the branch (git push origin feature/YourFeature)

Open a Pull Request

📝 License
This project is released under the MIT License.
Feel free to use, modify, and distribute!

pgsql
Copier
Modifier

Feel free to tweak any sections (e.g., install commands or example paths) to suit your workflow.
