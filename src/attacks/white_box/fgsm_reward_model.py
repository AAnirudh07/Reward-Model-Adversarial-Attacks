import torch
from attacks.base_attack import BaseAttack

class FGSMRewardModel(BaseAttack):
    """
    FGSM for reward models.
    
    Instead of using cross-entropy, this attack uses a custom loss:
    Loss = -reward, so that the adversary minimizes the reward score.
    
    Distance Measure: Linf
    
    Arguments:
        model (BaseModel): reward model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        batch_size (int): batch size for running model.inference in global mode. (Default: 8)
        average_over_dataset (bool): 
            If True, computes a single global loss (and gradient) by averaging the loss over all samples across batches.
            If False, processes one image at a time.
    """
    def __init__(self, model, eps=8/255, batch_size=8, average_over_dataset=False):
        super().__init__("FGSMRewardModel", model)
        self.eps = eps
        self.batch_size = batch_size
        self.average_over_dataset = average_over_dataset
        self.supported_mode = ["default"]

    def forward(self, images, labels):
        """
        Overridden forward method for attacking a reward model.
        """
        images = images.clone().detach().to(self.device)
        images.requires_grad = True
        num_images = images.shape[0]
        
        if self.average_over_dataset:
            total_loss = 0.0
            total_samples = 0

            for i in range(0, num_images, self.batch_size):
                batch_images = images[i: i + self.batch_size]
                batch_labels = labels[i: i + self.batch_size]

                batch_loss = -self.model.inference(batch_images, batch_labels).mean() * batch_images.shape[0]
                total_loss += batch_loss
                total_samples += batch_images.shape[0]

            global_loss = total_loss / total_samples
            grad = torch.autograd.grad(global_loss, images, retain_graph=False, create_graph=False)[0]
            adv_images = images + self.eps * grad.sign()
            adv_images = torch.clamp(adv_images, 0, 1).detach()
            return adv_images

        else:
            adv_images = []
            for i in range(num_images):
                image = images[i:i+1]  
                prompt = [labels[i]]   

                reward = self.model.inference(image, prompt)

                loss = -reward.mean()

                grad = torch.autograd.grad(loss, image, retain_graph=False, create_graph=False)[0]
                adv_image = image + self.eps * grad.sign()
                adv_image = torch.clamp(adv_image, 0, 1).detach()
                adv_images.append(adv_image)

            adv_images = torch.cat(adv_images, dim=0)
            return adv_images
