import argparse
import re

def check_target_model(value):
    pattern = r'^[^/]+/stable-diffusion-[123]$'
    if not re.match(pattern, value):
        raise argparse.ArgumentTypeError(
            "target_model_name must be in the format '<repo-owner>/stable-diffusion-[1|2|3]'.")
    return value

def check_reward_model(value):
    valid_versions = ["v1.0", "v2.0", "v2.1"]
    if value not in valid_versions:
        raise argparse.ArgumentTypeError(
            "reward_model_name must be one of: 'v1.0', 'v2.0', 'v2.1'.")
    return value

def check_dataset_name(value):
    if value not in ['hps', 'drawbench']:
        raise argparse.ArgumentTypeError(
            "dataset_name must be either 'hps' or 'drawbench'.")
    return value

def check_attack_name(value):
    if value not in ['gn', 'fgsm', 'pgd', 'spsa']:
        raise argparse.ArgumentTypeError(
            "attack_name must be one of: 'gn', 'fgsm', 'pgd', 'spsa'.")
    return value

def parse_args():
    parser = argparse.ArgumentParser(
        description="Argument parser for image generation and attack process."
    )
    
    # Models group
    models = parser.add_argument_group("models")
    models.add_argument("--target_model_name", type=check_target_model, required=True,
                        help="HuggingFace model ID in format <repo-owner>/stable-diffusion-[1|2|3]")
    models.add_argument("--reward_model_name", type=check_reward_model, required=True,
                        help="HPS reward model version: v1.0, v2.0, or v2.1")
    models.add_argument("--reward_threshold", type=float, default=70.0,
                        help="Minimum reward score for attack (default: 70.0)")

    # Datasets group
    datasets = parser.add_argument_group("datasets")
    datasets.add_argument("--dataset_name", type=check_dataset_name, required=True,
                        help="Dataset for generating preliminary images: 'hps' or 'drawbench'")
    datasets.add_argument("--num_samples_per_category", type=int, default=None,
                        help="Number of text prompts per category (default: 5 for hps, 2 for drawbench)")
    datasets.add_argument("--shuffle", action="store_true",
                        help="Shuffle prompts prior to sampling (default: False)")

    # Attack group
    attack = parser.add_argument_group("attack")
    attack.add_argument("--attack_name", type=check_attack_name, required=True,
                        help="Name of the perturbation attack: gn, fgsm, pgd, or spsa")

    # Misc group
    misc = parser.add_argument_group("misc")
    misc.add_argument("--inference_batch_size", type=int, default=4,
                        help="Batch size for target model inference (default: 4)")
    misc.add_argument("--no_save_image_results", dest="save_image_results", action="store_false",
                        help="Do not store images, prompts, and reward scores that pass threshold")
    misc.set_defaults(save_image_results=True)
    misc.add_argument("--use_cached_results", action="store_true", default=False,
                        help="Use stored images, prompts, and reward scores (default: False)")

    args = parser.parse_args()

    if args.num_samples_per_category is None:
        if args.dataset_name == "hps":
            args.num_samples_per_category = 5
        else:  # drawbench
            args.num_samples_per_category = 2

    return args
