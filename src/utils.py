import gc
import torch
from torch.utils.data import Dataset

class SampledDataset(Dataset):
    def __init__(self, prompts, images=None, transforms=None):
        self.data = [{"category": c, "prompt": p} for c, p in zip(prompts["category"], prompts["prompt"])]
        self.images = images
        self.transforms = transforms
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.images is not None:
            return self.transforms(self.images[idx]), self.data[idx]
        return self.data[idx]
    
def clear_cuda_memory_and_force_gc(force: bool = False):
    """
    Clears the CUDA memory cache and forces garbage collection if the allocated memory
    exceeds a certain threshold or if explicitly forced.

    Args:
        force (bool): If True, CUDA cache will be cleared and garbage collection
                      will be forced regardless of the memory threshold.
    """

    memory_allocated = torch.cuda.max_memory_reserved()
    memory_total = torch.cuda.get_device_properties("cuda").total_memory

    memory_threshold = memory_total * 0.7
    if memory_allocated > memory_threshold or force:
        torch.cuda.empty_cache()
        gc.collect()

def compute_reward_statistics(top_k_prompts, adv_rewards):
    """
    Compute reward statistics using both the original rewards from top_k_prompts 
    and the adversarial rewards in adv_rewards.

    Parameters:
        top_k_prompts (list): List of tuples (category, prompt, original_reward, image)
        adv_rewards (list): List of adversarial rewards corresponding to each prompt.
    """
    if not top_k_prompts or not adv_rewards:
        return {
            "average_original": 0.0,
            "average_adversarial": 0.0,
            "per_category_original": {},
            "per_category_adversarial": {}
        }

    original_rewards = [entry[2] for entry in top_k_prompts]
    avg_original = sum(original_rewards) / len(original_rewards)
    avg_adv = sum(adv_rewards) / len(adv_rewards)

    per_category_orig = {}
    per_category_adv = {}

    for (cat, _, orig_reward, _), adv_reward in zip(top_k_prompts, adv_rewards):
        per_category_orig.setdefault(cat, []).append(orig_reward)
        per_category_adv.setdefault(cat, []).append(adv_reward)

    per_category_orig_avg = {cat: sum(scores) / len(scores) for cat, scores in per_category_orig.items()}
    per_category_adv_avg = {cat: sum(scores) / len(scores) for cat, scores in per_category_adv.items()}

    return {
        "average_original": avg_original,
        "average_adversarial": avg_adv,
        "per_category_original": per_category_orig_avg,
        "per_category_adversarial": per_category_adv_avg
    }