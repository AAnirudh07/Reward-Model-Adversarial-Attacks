import re
import torch

from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusion3Pipeline  

from models.base_diffusion_model import BaseDiffusionModel

class StableDiffusionModel(BaseDiffusionModel):
    def __init__(self, model_path: str, offload_to_cpu: bool = False, resolution: int = None, **kwargs):
        """
        Note:
            model_path (str): Path to the Stable Diffusion model.
                              Must include 'stable-diffusion-1', 'stable-diffusion-2', or 'stable-diffusion-3' after '<repo-owner>/'
                              for simplicity.
        """

        # Load the model with float16 precision.
        # If your GPU supports torch.bfloat16 for lower memory usage with similar precision to FP32,
        # consider switching the torch_dtype accordingly.
        if "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = torch.float16        
        super().__init__(model_path, offload_to_cpu, resolution, **kwargs)

    def _get_diffusion_pipeline(self):
        version_tag = self.model_path.split("/")[-1].lower()
        if re.search(r'(stable-diffusion-1|v-?1)', version_tag):
            return StableDiffusionPipeline
        elif re.search(r'(stable-diffusion-2|v-?2)', version_tag):
            return DiffusionPipeline
        elif re.search(r'(stable-diffusion-3|v-?3)', version_tag):
            return StableDiffusion3Pipeline
        else:
            raise ValueError(
                "Model path must contain one of: 'stable-diffusion-1', 'stable-diffusion-2', or 'stable-diffusion-3'."
            )