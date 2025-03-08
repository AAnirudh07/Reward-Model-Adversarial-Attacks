import os
import re
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

from models import ModelFactory
from datasets import DatasetFactory
from datasets.round_robin_sampler import RoundRobinSampler

from utils import SampledDataset, clear_cuda_memory_and_force_gc
from args import parse_model_args

def generate_images(generate_image_args):
    print(generate_image_args)
    kwargs = {}

    if re.search(r'(stable-diffusion-2|v-?2)', generate_image_args.target_model_name):
        kwargs = {
            "resolution": 512,
        }
    if re.search(r'(stable-diffusion-3|v-?3)', generate_image_args.target_model_name):
        kwargs = {
            "resolution": 1024,
            "offload_to_cpu": True,
            "text_encoder_3": None,
            "tokenizer_3": None,
            "token": "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"

        }

    model = ModelFactory.create_model(
        model_type="sd",
        model_path=generate_image_args.target_model_name,
        **kwargs,
    )

    dataset = DatasetFactory.create_dataset(
        dataset_type=generate_image_args.dataset_name,
    )

    # Generate twice as many images as the number of samples per category for safety
    num_images_to_gen = 2 * generate_image_args.num_samples_per_category * dataset.num_categories()
    dataset_loader = DataLoader(
        dataset,
        batch_size=num_images_to_gen,
        sampler=RoundRobinSampler(dataset) if generate_image_args.shuffle else None,
    )

    prompts = next(iter(dataset_loader))
    sampled_dataset = SampledDataset(prompts)
    sampled_dataset_loader = DataLoader(sampled_dataset, batch_size=generate_image_args.inference_batch_size, shuffle=False)

    final_images = []
    final_prompts = []
    total_batches = len(sampled_dataset_loader)
    pbar = tqdm(total=total_batches, desc="Generating images from prompts")

    for batch in sampled_dataset_loader:
        prompts = batch["prompt"]
        categories = batch["category"]
        images = model.inference(inputs=prompts)
        final_images.extend(images)
        final_prompts.extend([(category, prompt) for category, prompt in zip(categories, prompts)])
        pbar.update(1)
    pbar.close()

    if generate_image_args.save_image_results:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = os.path.join(f"outputs/{generate_image_args.target_model_name.split('/')[1]}/{generate_image_args.dataset_name}/{timestamp}/")
        os.makedirs(output_dir)

        prompts_file = os.path.join(output_dir, "prompts.txt")
        with open(prompts_file, "w") as pf:
            for idx, (img, prompt) in enumerate(zip(final_images, final_prompts)):
                image_filename = os.path.join(output_dir, f"image_{idx}.png")
                img.save(image_filename)
                pf.write(f"Image {idx}: {prompt}\n")


    clear_cuda_memory_and_force_gc(force=True)
