import torch
from models.base_diffusion_model import BaseDiffusionModel
from diffusers import FluxPipeline

class FluxDiffusionModel(BaseDiffusionModel):
    def __init__(self, model_path: str, offload_to_cpu: bool = True, resolution: int = 512, **kwargs):
        if "torch_dtype" not in kwargs:
            kwargs["torch_dtype"] = torch.float16        
        super().__init__(model_path, offload_to_cpu, resolution, **kwargs)
        
        self.diffusion_pipeline = FluxDiffusionModel