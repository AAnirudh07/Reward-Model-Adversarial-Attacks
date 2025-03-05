import torch
from attacks.base_attack import BaseAttack

class GNRewardModel(BaseAttack):
    r"""
    Gaussian Noise attack for reward models.

    Arguments:
        model (BaseModel): reward model to attack.
        std (float): standard deviation of the Gaussian noise (Default: 0.1).
    """

    def __init__(self, model, std=0.1):
        super().__init__("GNReward", model)
        self.std = std
        self.supported_mode = ["default"]

    def forward(self, images, labels=None):
        """
        Overridden forward method for attacking a reward model.
        """
        images = images.clone().detach().to(self.device)
        adv_images = images + self.std * torch.randn_like(images)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        return adv_images
