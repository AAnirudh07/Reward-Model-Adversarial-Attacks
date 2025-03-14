import torch
from typing import List, Optional
from models.base_model import BaseModel
from models.error import ModelLoadingError, InferenceError
from diffusers import DiffusionPipeline

class BaseDiffusionModel(BaseModel):
    def __init__(self, model_path: str, offload_to_cpu: bool = False, resolution: int = None, **kwargs):
        """
        Args:
            model_path (str): Path or repository ID of the diffusion model checkpoint.
        """
        self.seed = 42

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.offload_to_cpu = offload_to_cpu
        self.resolution = resolution
        self.kwargs = kwargs
        
        self.diffusion_pipeline = self._get_diffusion_pipeline()
        self.load_model()

    def _get_diffusion_pipeline(self):
        """ Subclasses should override this to return the correct pipeline. """
        return DiffusionPipeline

    def load_model(self):
        try:
            self.model = self.diffusion_pipeline.from_pretrained(
                self.model_path,
                **self.kwargs
            ).to(self.device)
            if self.offload_to_cpu:
                self.model.enable_model_cpu_offload()

        except MemoryError as e:
            if hasattr(self, "model"):
                del self.model
                torch.cuda.empty_cache()
            raise ModelLoadingError(f"Memory error occurred while loading the model. Consider using a smaller model: {e}")
        except FileNotFoundError as e:
            raise ModelLoadingError(f"Model checkpoint not found at '{self.model_path}'.") from e
        except Exception as e:
            raise ModelLoadingError(f"Failed to load diffusion model: {e}") from e

    def inference(
        self, inputs: List[str], captions: Optional[List[str]] = None
    ):
        """
        Runs inference on a batch of prompts.
        Returns a batch of images corresponding to the prompts.
        """
        if not isinstance(inputs, list) or not all(isinstance(c, str) for c in inputs):
            raise TypeError("Expected 'inputs' to be a list of strings.")

        try:
            # Create one generator per prompt to ensure reproducibility
            generators = [
                torch.Generator(self.device).manual_seed(self.seed) for _ in range(len(inputs))
            ]
            if self.resolution:
                images = self.model(
                    prompt=inputs, generator=generators,
                    height=self.resolution, width=self.resolution # use 1:1 aspect ratio
                ).images
                return images
            else:
                images = self.model(
                    prompt=inputs, generator=generators,
                ).images
                return images

        except Exception as e:
            raise InferenceError(f"Inference failed: {e}")