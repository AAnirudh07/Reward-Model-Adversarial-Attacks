import torch
import torch.nn as nn
from attacks.base_attack import BaseAttack

class PGDRewardModel(BaseAttack):
    """
    PGD for reward models with global loss averaging over all batches.
    
    Instead of using cross-entropy loss, this attack uses a custom loss:
    Loss = -reward, so that the adversary minimizes the reward score.
    
    In this variant, the loss is computed over the entire dataset (split into mini-batches)
    and the gradient is averaged over all samples.
    
    Distance Measure : Linf
    
    Arguments:
        model (nn.Module): reward model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of PGD steps. (Default: 10)
        random_start (bool): if True, initializes the adversarial example with a random perturbation.
        batch_size (int): batch size for computing the global loss. (Default: 8)
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
        Overridden forward method for attacking a reward model using a global averaged loss.
        """
        images = images.clone().detach().to(self.device)
        adv_images = images.clone().detach()
        num_images = images.shape[0]

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, 0, 1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            total_loss = 0.0
            total_samples = 0
            for i in range(0, num_images, self.batch_size):
                batch_images = adv_images[i: i + self.batch_size]
                batch_labels = labels[i: i + self.batch_size]

                total_loss += -sum(self.model.inference(batch_images, batch_labels))
                total_samples += batch_images.shape[0]

            global_loss = total_loss / total_samples

            grad = torch.autograd.grad(global_loss, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, 0, 1).detach()

        return adv_images
