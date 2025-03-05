import torch
import torch.nn as nn

from torchattacks.attack import Attack

class FGSMRewardModel(Attack):
    """
    FGSM for reward models.
    
    Instead of using the cross-entropy loss, this attack uses a custom loss:
    Loss = -reward, so that the adversary minimizes the reward score.
    
    Distance Measure : Linf
    
    Arguments:
        model (BaseModel): reward model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
    """
    def __init__(self, model, eps=8/255):
        super().__init__("FGSMRewardModel", model)
        self.eps = eps
        self.supported_mode = ["default"]

    def forward(self, images):
        """
        Overridden forward method for attacking a reward model.
        """

        images = images.clone().detach().to(self.device)
        images.requires_grad = True

        reward = self.model.inference(images)
        cost = -reward

        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
