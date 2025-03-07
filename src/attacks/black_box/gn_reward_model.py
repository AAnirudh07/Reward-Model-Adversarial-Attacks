import torch

from attacks.base_attack import BaseAttack
from datasets.image_prompt_dataset import ImagePromptDataset

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

    def forward(self, images, labels):
        """
        Overridden forward method for attacking a reward model.
        """

        dataset = ImagePromptDataset(
            image_list=images, prompt_list=labels,
            image_transform_function=self.model.preprocess_function,
            text_tokenizer_function=self.model.tokenizer
        )

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=len(images), shuffle=False
        )

        images, _ = next(iter(dataloader))
        images = images.clone().detach().to(self.device)
        adv_images = images + self.std * torch.randn_like(images)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        return adv_images
