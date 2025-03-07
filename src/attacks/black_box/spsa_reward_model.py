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

    This attack approximates the gradient of the reward output with respect to the input
    image using SPSA (Simultaneous Perturbation Stochastic Approximation) and updates the input
    to reduce the reward score.

    Distance Measure : L_infinity

    Arguments:
        model (nn.Module): Reward model to attack. It should take an image tensor and output a scalar reward.
        eps (float): maximum perturbation. (Default: 8/255)
        delta (float): smoothing parameter for gradient approximation. (Default: 0.01)
        lr (float): learning rate for the optimizer. (Default: 0.01)
        nb_iter (int): number of attack iterations. (Default: 1)
        nb_sample (int): number of samples for SPSA gradient approximation. (Default: 128)
        max_batch_size (int): maximum batch size for gradient estimation. (Default: 64)
    """
    def __init__(self, model, eps=8/255, delta=0.01, lr=0.01, nb_iter=1, nb_sample=128, max_batch_size=64, dataset_batch_size=8):
        super().__init__("SPSARewardModel", model)
        self.eps = eps
        self.delta = delta
        self.lr = lr
        self.nb_iter = nb_iter
        self.nb_sample = nb_sample
        self.max_batch_size = max_batch_size
        self.dataset_batch_size = dataset_batch_size
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
        grad = torch.zeros_like(images)
        images = torch.unsqueeze(images, 0)
        labels = torch.unsqueeze(labels, 0)

        images = images.expand(max_batch_size, *images.shape[1:]).contiguous()
        labels = labels.expand(max_batch_size, *labels.shape[1:]).contiguous()

        v = torch.empty_like(images[:, :1, ...])
        for batch_size in self._get_batch_sizes(nb_sample, max_batch_size):
            x_ = images[:batch_size]
            y_ = labels[:batch_size]
            vb = v[:batch_size]
            vb = vb.bernoulli_().mul_(2.0).sub_(1.0)
            v_ = vb.expand_as(x_).contiguous()
            x_shape = x_.shape
            x_ = x_.view(-1, *images.shape[2:])
            y_ = y_.view(-1, *labels.shape[2:])
            v_ = v_.view(-1, *v.shape[2:])
            df = self.loss(x_ + delta * v_, y_) - self.loss(x_ - delta * v_, y_)
            df = df.view(-1, *[1 for _ in v_.shape[1:]])
            grad_ = df / (2.0 * delta * v_)
            grad_ = grad_.view(x_shape)
            grad_ = grad_.sum(dim=0, keepdim=False)
            grad += grad_

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