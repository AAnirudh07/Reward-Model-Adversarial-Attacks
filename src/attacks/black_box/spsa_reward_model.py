"""
Code is modified from https://github.com/BorealisAI/advertorch/blob/master/advertorch/attacks/spsa.py
"""

import torch
from torch.utils.data import DataLoader

from attacks.base_attack import BaseAttack
from datasets.image_prompt_dataset import ImagePromptDataset

class SPSARewardModel(BaseAttack):
    """
    SPSA attack for reward models.

    Distance Measure : L_inf

    Arguments:
        model (nn.Module): Reward model to attack. It should take an image tensor and output a scalar reward.
        eps (float): maximum perturbation. (Default: 0.3)
        delta (float): smoothing parameter for gradient approximation. (Default: 0.01)
        lr (float): learning rate for the optimizer. (Default: 0.01)
        nb_iter (int): number of attack iterations. (Default: 1)
        nb_sample (int): number of samples for SPSA gradient approximation. (Default: 128)
        max_batch_size (int): maximum batch size for gradient estimation. (Default: 64)
        batch_size (int): mini-batch size for processing inputs from the dataset. (Default: 8)
    """
    def __init__(self, model, eps=0.3, delta=0.01, lr=0.01, nb_iter=1, nb_sample=128, max_batch_size=64, batch_size=8):
        super().__init__("SPSARewardModel", model)
        self.eps = eps
        self.delta = delta
        self.lr = lr
        self.nb_iter = nb_iter
        self.nb_sample = nb_sample
        self.max_batch_size = max_batch_size
        self.dataset_batch_size = batch_size
        self.supported_mode = ["default"]

    def forward(self, images, labels):
        dataset = ImagePromptDataset(
            image_list=images, prompt_list=labels,
            image_transform_function=self.model.preprocess_function,
            text_tokenizer_function=self.model.tokenizer
        )
        dataloader = DataLoader(dataset, batch_size=self.dataset_batch_size, shuffle=False)

        adv_images_list = []
        for images, labels in dataloader:
            images = images.clone().detach().to(self.device)
            labels = labels.clone().detach().to(self.device)
            adv = self.spsa_perturb(images, labels)
            adv_images_list.append(adv)
        return torch.cat(adv_images_list, dim=0)

    def loss(self, images, labels):
        reward = self.model.inference(images, labels)
        reward = torch.tensor(reward, device=images.device)
        return -reward.mean()

    def linf_clamp_(self, dx, x, eps):
        dx_clamped = torch.clamp(dx, min=-eps, max=eps)
        x_adv = torch.clamp(x + dx_clamped, min=0, max=1)
        # In-place update for proper optimizer tracking.
        dx += x_adv - x - dx
        return dx

    def _get_batch_sizes(self, n, max_batch_size):
        batches = [max_batch_size for _ in range(n // max_batch_size)]
        if n % max_batch_size > 0:
            batches.append(n % max_batch_size)
        return batches

    @torch.no_grad()
    def spsa_grad(self, images, labels, delta, nb_sample, max_batch_size):
        # images shape: (B, C, H, W)
        grad = torch.zeros_like(images)
        B = images.shape[0]

        images = images.unsqueeze(1)   # (B, 1, C, H, W)
        labels = labels.unsqueeze(1)   # (B, 1, P)

        images = images.expand(B, max_batch_size, *images.shape[2:]).contiguous()  # (B, max_batch_size, C, H, W)
        labels = labels.expand(B, max_batch_size, *labels.shape[2:]).contiguous()

        v = torch.empty_like(images[:, :, :1, ...])  # (B, max_batch_size, 1, H, W)
        for current_batch in self._get_batch_sizes(nb_sample, max_batch_size):
            x_batch = images[:, :current_batch].contiguous()  # (B, current_batch, C, H, W)
            y_batch = labels[:, :current_batch].contiguous()    # (B, current_batch, P)
            v_batch = v[:, :current_batch]
            v_batch.bernoulli_().mul_(2.0).sub_(1.0)
            v_batch_exp = v_batch.expand_as(x_batch).contiguous()  # (B, current_batch, C, H, W)

            B_curr, bs, C, H, W = x_batch.shape
            x_batch_reshaped = x_batch.view(B_curr * bs, C, H, W)
            y_batch_reshaped = y_batch.view(B_curr * bs, -1)
            v_batch_reshaped = v_batch_exp.view(B_curr * bs, C, H, W)

            df = self.loss(x_batch_reshaped + delta * v_batch_reshaped, y_batch_reshaped) \
                 - self.loss(x_batch_reshaped - delta * v_batch_reshaped, y_batch_reshaped)
            df = df.view(-1, *([1] * (v_batch_reshaped.dim()-1)))
            grad_batch = (df / (2.0 * delta)) * v_batch_reshaped # equivalent to original code as each element of v_batch_reshaped is +-1
            grad_batch = grad_batch.view(B_curr, bs, C, H, W)
            grad += grad_batch.sum(dim=1)

        grad /= nb_sample
        return grad

    def spsa_perturb(self, x, y):
        dx = torch.zeros_like(x)
        dx.grad = torch.zeros_like(dx)
        optimizer = torch.optim.Adam([dx], lr=self.lr)
        for _ in range(self.nb_iter):
            optimizer.zero_grad()
            dx.grad = self.spsa_grad(x + dx, y, self.delta, self.nb_sample, self.max_batch_size)
            optimizer.step()
            dx = self.linf_clamp_(dx, x, self.eps)
        x_adv = x + dx
        return x_adv