import os
import ast
from PIL import Image
from torch.utils.data import DataLoader
    
from models.model_factory import ModelFactory
from utils import SampledDataset, clear_cuda_memory_and_force_gc, compute_reward_statistics, numerical_key

def run_transfer_test(run_transfer_test_args):
    print(run_transfer_test_args)

    reward_model = ModelFactory.create_model(
        model_type=run_transfer_test_args.reward_model_name, 
        model_path="hpc.pt" if run_transfer_test_args.reward_model_name == "hpsv1" else "HPS_v2_compressed.pt",
    )

    original_image_directory = run_transfer_test_args.original_images_path
    prompts_file = os.path.join(original_image_directory, "prompts.txt")
    original_prompts = {"category": [], "prompt": []}
    with open(prompts_file, "r") as pf:
        for line in pf:
            content = line.split(": ", 1)[1].strip()
            if content.startswith("(") and content.endswith(")"):
                category, prompt = ast.literal_eval(content)
                original_prompts["category"].append(category)
                original_prompts["prompt"].append(prompt)    
    
    generated_image_directory = run_transfer_test_args.generated_images_path
    generated_prompts_file = os.path.join(generated_image_directory, "prompts.txt")
    generated_prompts = {"category": [], "prompt": [], "orig_r": [], "adv_r": []}
    with open(generated_prompts_file, "r") as pf:
        for line in pf:
            parts = line.split(": ", 1)
            content = parts[1].strip()
            if content.startswith("(") and content.endswith(")"):
                category, prompt, orig_r, adv_r = ast.literal_eval(content)
                generated_prompts["category"].append(category)
                generated_prompts["prompt"].append(prompt)

    original_image_files = sorted([f for f in os.listdir(original_image_directory) if f.endswith(".png")], key=numerical_key) 
    original_images = [Image.open(os.path.join(original_image_directory, img_file)) for img_file in original_image_files]

    generated_image_files = sorted([f for f in os.listdir(generated_image_directory) if f.endswith(".png")], key=numerical_key) 
    generated_images = [Image.open(os.path.join(generated_image_directory, img_file)) for img_file in generated_image_files]

    original_mapping = {}
    for idx, (cat, prompt) in enumerate(zip(original_prompts["category"], original_prompts["prompt"])):
        original_mapping[(cat, prompt)] = original_images[idx]

    corresponding_original_images = []
    for gen_cat, gen_prompt in zip(generated_prompts["category"], generated_prompts["prompt"]):
        orig_img = original_mapping.get((gen_cat, gen_prompt))
        corresponding_original_images.append(orig_img)

    # Get original reward scores    
    original_reward_scores = []
    original_dataset = SampledDataset(
        prompts=original_prompts, images=original_images, 
        transforms=reward_model.preprocess_function
    )
    dataloader = DataLoader(original_dataset, batch_size=8, shuffle=False)
    for batch in dataloader:
        images, prompts = batch
        prompt_texts = prompts["prompt"]
        original_reward_scores.extend(reward_model.inference(inputs=images, captions=prompt_texts))
    generated_prompts["orig_r"] = original_reward_scores

    # Get generated reward scores 
    generated_reward_scores = []
    generated_dataset = SampledDataset(
        prompts=generated_prompts, images=generated_images, 
        transforms=reward_model.preprocess_function
    )  
    dataloader = DataLoader(generated_dataset, batch_size=8, shuffle=False)
    for batch in dataloader:
        images, prompts = batch
        prompt_texts = prompts["prompt"]
        generated_reward_scores.extend(reward_model.inference(inputs=images, captions=prompt_texts))
    generated_prompts["adv_r"] = generated_reward_scores

    stats = compute_reward_statistics(generated_prompts, generated_prompts["adv_r"])
    print("\n" + "=" * 40)
    print("Overall Reward Statistics:")
    print(f"  Original: {stats['average_original']} | Adversarial: {stats['average_adversarial']}\n")

    print("Per-Category Comparison:")
    all_categories = set(stats["per_category_original"].keys()).union(stats["per_category_adversarial"].keys())
    for cat in all_categories:
        orig = stats["per_category_original"].get(cat, 0)
        adv = stats["per_category_adversarial"].get(cat, 0)
        print(f"  {cat}: Original = {orig} | Adversarial = {adv}")
    print("=" * 40)

    clear_cuda_memory_and_force_gc(force=True)
    