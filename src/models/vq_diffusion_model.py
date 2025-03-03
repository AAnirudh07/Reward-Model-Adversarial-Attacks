from models.base_diffusion_model import BaseDiffusionModel
from diffusers import VQDiffusionPipeline

class VQDiffusionModel(BaseDiffusionModel):
    def __init__(self, model_path: str):
        super().__init__(model_path)
        
        self.diffusion_pipeline = VQDiffusionPipeline