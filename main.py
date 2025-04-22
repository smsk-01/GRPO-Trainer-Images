from rewards import accuracy_reward_coord
from GRPOImagesTrainer import PersonalizedGRPOTrainer 
import torch
from GRPOImagesTrainer import Qwen2_5_VLForConditionalGenerationWithLogits
from transformers import AutoProcessor
from datasets import load_dataset
from trl import GRPOConfig




if __name__ == '__main__':
    IMAGES_ROOT = 'image_root'
    JSON_PATH   = 'json_file.json'

    dataset = load_dataset('json', data_files=JSON_PATH, split='train')

    processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')
    image_processor = processor.image_processor
    tokenizer      = processor.tokenizer


    config = GRPOConfig(
        output_dir='Qwen2.5-VL-3B-GRPO',
        per_device_train_batch_size=4,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        num_generations=4,
    )

    model = Qwen2_5_VLForConditionalGenerationWithLogits.from_pretrained(
        'Qwen/Qwen2.5-VL-3B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )

    trainer = PersonalizedGRPOTrainer(
        model=model,
        args=config,
        reward_funcs=[accuracy_reward_coord],
        train_dataset=dataset,
        processing_class=processor.tokenizer,
        image_processor=image_processor,
        images_root=IMAGES_ROOT,
    )

    trainer.train()