import torch
from torch.utils.data import DataLoader
from attacks.base_attack import BaseAttack
from datasets.image_prompt_dataset import ImagePromptDataset

class PGDRewardModel(BaseAttack):
    """
    PGD for reward models using global loss averaging over the entire dataset.

    Instead of using cross-entropy loss, this attack uses a custom loss:
    Loss = -reward, so that the adversary minimizes the reward score.

    The entire dataset is loaded into memory, and during each PGD step we iterate
    over mini-batches of the current adversarial images by simple slicing.

    Distance Measure: Linf

    Arguments:
        model (nn.Module): reward model to attack.
        eps (float): maximum perturbation (Default: 8/255).
        alpha (float): step size (Default: 2/255).
        steps (int): number of PGD steps (Default: 10).
        random_start (bool): if True, initializes adversarial examples with a random perturbation.
        batch_size (int): mini-batch size for computing the global loss (Default: 8).
    """
    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, random_start=True, batch_size=8):
        super().__init__("PGDRewardModel", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.batch_size = batch_size
        self.supported_mode = ["default"]

    def forward(self, images, labels):
        """
        Overridden forward method for attacking a reward model.
        """
        dataset = ImagePromptDataset(
            image_list=images, prompt_list=labels, 
            image_transform_function=self.model.preprocess_function,
            text_tokenizer_function=None
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        images_list = []
        prompts_list = []

        for imgs, prompts in loader:
            images_list.append(imgs)
            prompts_list.extend(prompts)

        all_images = torch.cat(images_list, dim=0).to(self.device)
        adv_images = all_images.clone().detach()
        num_images = all_images.shape[0]

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, 0, 1).detach()

        # PGD steps: process the global adversarial images in mini-batches using slicing.
        for _ in range(self.steps):
            total_loss = torch.tensor(0.0, device=self.device)
            total_samples = 0

            for i in range(0, num_images, self.batch_size):
                batch_images = adv_images[i: i + self.batch_size]
                batch_prompts = prompts_list[i: i + self.batch_size]
                batch_images.requires_grad_() 

                reward = self.model.inference_with_grad(batch_images, batch_prompts)
                total_loss += -reward.sum() 
                total_samples += batch_images.shape[0]

            global_loss = total_loss / total_samples
            grad = torch.autograd.grad(global_loss, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - all_images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(all_images + delta, 0, 1).detach()

        return adv_images