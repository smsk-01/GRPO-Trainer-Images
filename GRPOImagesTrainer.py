import os
import re
from datetime import datetime
from PIL import Image
from datasets import load_dataset
import torch
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from trl import GRPOTrainer, GRPOConfig


class PersonalizedGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, image_processor, images_root, **kwargs):
        assert image_processor is not None
        assert images_root is not None
        self.image_processor = image_processor
        self.images_root    = images_root
        super().__init__(*args, **kwargs)
        # override the data_collator
        self.data_collator = self._image_data_collator

    def _image_data_collator(self, features: list[dict]) -> list[dict]:
        # ---- load and preprocess images ----
        imgs = []
        for f in features:
            path = os.path.join(self.images_root, f['img_filename'])
            imgs.append(Image.open(path).convert('RGB'))
        pixel_batch = self.image_processor(images=imgs, return_tensors="pt")
        pixel_values = pixel_batch.pixel_values
        scales       = [(img.width, img.height) for img in imgs]

        # ---- collect text/bboxes ----
        prompts = [f['instruction'] for f in features]
        bboxes  = [f['bbox']         for f in features]

        # ---- now build a list of per-sample dicts ----
        batch: list[dict] = []
        for i in range(len(features)):
            batch.append({
                'pixel_values':   pixel_values[i],   # (C,H,W) tensor
                'prompt':         prompts[i],        # string
                'solution':       bboxes[i],         # list of ints
                'scales':         scales[i],         # (width,height)
                # you *could* also include input_ids / attention_mask here
                # if you want to skip re-tokenization downstream, e.g.:
                # 'input_ids':      text_inputs.input_ids[i],
                # 'attention_mask': text_inputs.attention_mask[i],
            })
        return batch

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

class Qwen2_5_VLForConditionalGenerationWithLogits(Qwen2_5_VLForConditionalGeneration):
    def forward(
        self,
        *args,
        logits_to_keep: int | None = None,  # accept and ignore
        **kwargs,
    ):
        # drop logits_to_keep and pass everything else on
        return super().forward(*args, **kwargs)