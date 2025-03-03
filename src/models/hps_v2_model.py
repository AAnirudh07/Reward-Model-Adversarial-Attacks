from typing import List

import hpsv2
import PIL

from models.base_model import BaseModel
from models.error import ModelLoadingError, InferenceError

class HPSv2Model(BaseModel):
    def __init__(self, model_path: str):
        """
        Args:
            model_path (str): Path to the HPSv2 model checkpoint. Must be
            either 'v2.0' or 'v2.1'.
        """
        if model_path not in ["v2.0", "v2.1"]:
            raise ValueError("Expected 'model_path' to be either 'v2.0' or 'v2.1'.")
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        try:
            temp_image = PIL.Image.new("RGB", (256, 256), color=(255, 255, 255))
            _ = hpsv2.score(temp_image, '<prompt>', hps_version="v2.0") # Also caches the model

        except FileNotFoundError as e:
            raise ModelLoadingError(f"Model checkpoint not found at '{self.model_path}'.") from e
        except Exception as e:
            raise ModelLoadingError(f"Error loading model: {e}") from e

    def inference(self, inputs: List[PIL.Image], captions: List[str]) -> List[float]:
        """
        Runs inference on a batch of images and corresponding captions.
        Returns a batch of reward scores.
        """
        if not isinstance(inputs, list) or not all(isinstance(i, PIL.Image.Image) for i in inputs):
            raise TypeError("Expected 'inputs' to be a list of PIL.Image objects.")
        if not isinstance(captions, list) or not all(isinstance(c, str) for c in captions):
            raise TypeError("Expected 'captions' to be a list of strings.")
        if len(inputs) != len(captions):
            raise ValueError("Number of 'inputs' and 'captions' must match.")
        
        try:
            similarity_scores = []
            for i in range(len(inputs)):
                reward = hpsv2.score(inputs[i], captions[i], hps_version=self.model_path)
                similarity_scores.append(reward[0] * 100)
            return similarity_scores

        except Exception as e:
            raise InferenceError(f"Inference failed: {e}") from e