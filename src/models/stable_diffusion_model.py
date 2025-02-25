import re
import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusion3Pipeline  
from typing import List, Optional
from models.base_model import BaseModel
from models.error import ModelLoadingError, InferenceError

class StableDiffusionModel(BaseModel):
    def __init__(self, model_path: str):
        """
        Args:
            model_path (str): Path to the Stable Diffusion model.
                              Must include 'stable-diffusion-1', 'stable-diffusion-2', or 'stable-diffusion-3' after '<repo-owner>/'
                              for simplicity.
        """
        self.seed = 42

        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):

        version_tag = self.model_path.split("/")[-1].lower()

        if re.search(r'(stable-diffusion-1|v-?1)', version_tag):
            pipeline_class = StableDiffusionPipeline
        elif re.search(r'(stable-diffusion-2|v-?2)', version_tag):
            pipeline_class = DiffusionPipeline
        elif re.search(r'(stable-diffusion-3|v-?3)', version_tag):
            pipeline_class = StableDiffusion3Pipeline
        else:
            raise ModelLoadingError(
                "Model path must contain one of: 'stable-diffusion-1', 'stable-diffusion-2', or 'stable-diffusion-3'."
            )
        
        try:
            # Load the model with float16 precision.
            # If your GPU supports torch.bfloat16 for lower memory usage with similar precision to FP32,
            # consider switching the torch_dtype accordingly.
            self.diffusion_pipeline = pipeline_class.from_pretrained(
                self.model_path, torch_dtype=torch.float16
            ).to(self.device)
            
        except MemoryError as e:
            # Clean up and clear GPU memory if a MemoryError occurs
            if hasattr(self, "model"):
                del self.model
            torch.cuda.empty_cache()
            raise ModelLoadingError(f"Memory error occurred while loading the model. Consider using a smaller model: {e}") from e        
        except Exception as e:
            raise ModelLoadingError(f"Failed to load Stable Diffusion model: {e}") from e

    def inference(
        self, inputs: List[str], captions: Optional[List[str]] = None
    ) -> list[torch.Tensor]:
        """
        Runs inference on a batch of prompts.
        Returns a batch of images corresponding to the prompts.
        """
        if not isinstance(inputs, list) or not all(isinstance(c, str) for c in inputs):
            raise TypeError("Expected 'inputs' to be a list of strings.")

        try:
            # Create one generator per prompt to ensure reproducibility
            generators = [
                torch.Generator("cuda").manual_seed(self.seed) for _ in range(len(inputs))
            ]
            images = self.diffusion_pipeline(prompt=inputs, generator=generators).images
            return images
        
        except Exception as e:
            raise InferenceError(f"Inference failed: {e}")