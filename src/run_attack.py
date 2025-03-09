import os
import ast  
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as T

from models.model_factory import ModelFactory
from attacks.attack_factory import AttackFactory
from utils import SampledDataset, clear_cuda_memory_and_force_gc, compute_reward_statistics, numerical_key

reward_model = ModelFactory.create_model(
    model_type="hpsv2",   
    model_path="HPS_v2_compressed.pt",
)
to_pil = T.ToPILImage()

def run_attack_rank_model(run_attack_args):
    print(run_attack_args)

    image_directory = run_attack_args.saved_images_path
    prompts_file = os.path.join(image_directory, "prompts.txt")
    prompts = {"category": [], "prompt": []}

    with open(prompts_file, "r") as pf:
        for line in pf:
            content = line.split(": ", 1)[1].strip()
            if content.startswith("(") and content.endswith(")"):
                category, prompt = ast.literal_eval(content)
                prompts["category"].append(category)
                prompts["prompt"].append(prompt)

    image_files = sorted([f for f in os.listdir(image_directory) if f.endswith(".png")], key=numerical_key) 
    original_images = [Image.open(os.path.join(image_directory, img_file)) for img_file in image_files]

    dataset = SampledDataset(
        prompts=prompts, images=original_images, 
        transforms=reward_model.preprocess_function
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    all_results = []  # will store tuples: (category, prompt, reward_score)
    total_batches = len(dataloader)
    pbar = tqdm(total=total_batches, desc="Ranking images and prompts")

    for batch in dataloader:
        images, prompts = batch
        categories  = prompts["category"]
        prompt_texts = prompts["prompt"]

        reward_scores = reward_model.inference(inputs=images, captions=prompt_texts)
        for cat, pr, score, image in zip(categories, prompt_texts, reward_scores, images):
            all_results.append((cat, pr, score, to_pil(image.cpu())))
        pbar.update(1)
    pbar.close()  

    filtered_results = [entry for entry in all_results if entry[2] >= run_attack_args.reward_threshold]
    ranked_results = sorted(filtered_results, key=lambda x: x[2], reverse=True)
    k = run_attack_args.num_samples_per_category * 4 if run_attack_args.dataset_name == "hps" else 11
    top_k_prompts = ranked_results[:k]
    
    print("Top prompts:")
    for idx, (cat, pr, score, _) in enumerate(top_k_prompts):
        print(f"Image {idx}: ({cat}, {pr}) with score {score}")
    return top_k_prompts, reward_model

def run_attack_reward_model(run_attack_args, top_k_prompts, reward_model):
    attack = AttackFactory.create_attack(
        attack_type=run_attack_args.attack_name,
        model=reward_model,
        batch_size=run_attack_args.attack_batch_size
    )

    prompts = [(cat, pr) for cat, pr, _, _ in top_k_prompts]
    prompts_only = [pr for _, pr, _, _ in top_k_prompts]
    original_images = [image for _, _, _, image in top_k_prompts]
    original_rewards = [score for _, _, score, _ in top_k_prompts]

    adv_images = attack(
        inputs=original_images,
        labels=prompts,
    )
    adv_rewards = reward_model.inference(inputs=adv_images, captions=prompts_only)

    if run_attack_args.save_image_results:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        parts = run_attack_args.saved_images_path.split(os.sep)
        output_dir = f"outputs/{parts[1]}/{run_attack_args.dataset_name}_adversarial/{run_attack_args.attack_name}/{timestamp}/"
        os.makedirs(output_dir, exist_ok=True)

        prompts_file_path = os.path.join(output_dir, "prompts.txt")
        with open(prompts_file_path, "w") as pf:
            for idx, (pr, orig_r, adv_r, adv_img) in enumerate(zip(prompts, original_rewards, adv_rewards, adv_images)):
                image_filename = os.path.join(output_dir, f"image_{idx}.png")
                pil_img = to_pil(adv_img.cpu())
                pil_img.save(image_filename)
                pf.write(f"Image {idx}: ({repr(pr[0])}, {repr(pr[1])}, {orig_r}, {adv_r})\n")

    stats = compute_reward_statistics(top_k_prompts, adv_rewards)
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