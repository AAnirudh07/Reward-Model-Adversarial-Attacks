import torch
from torch.utils.data import DataLoader
from attacks.base_attack import BaseAttack
from datasets.image_prompt_dataset import ImagePromptDataset

class FGSMRewardModel(BaseAttack):
    """
    FGSM for reward models.

    Instead of using cross-entropy, this attack uses a custom loss:
    Loss = -reward, so that the adversary minimizes the reward score.

    Distance Measure: Linf

    Arguments:
        model (BaseModel): reward model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        batch_size (int): batch size for processing images via DataLoader.
    """
    def __init__(self, model, eps=8/255, batch_size=1):
        super().__init__("FGSMRewardModel", model)
        self.eps = eps
        self.batch_size = batch_size
        self.supported_mode = ["default"]

    def forward(self, images, labels):
        """
        Overridden forward method for attacking a reward model using a DataLoader.
        """
        dataset = ImagePromptDataset(images, labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        adv_images_list = []

        for images, labels in loader:
            images = images.clone().detach().to(self.device)
            images.requires_grad = True

            reward = self.model.inference_with_grad(images, labels)
            loss = -reward.mean()
            grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
            adv_batch = images + self.eps * grad.sign()
            adv_batch = torch.clamp(adv_batch, 0, 1).detach()
            adv_images_list.append(adv_batch)

        adv_images = torch.cat(adv_images_list, dim=0)
        return adv_images
