from typing import List, Union

import torch
import PIL
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

from models.base_model import BaseModel
from models.error import ModelLoadingError, InferenceError

class HPSv2Model(BaseModel):
    def __init__(self, model_path: str):
        """
        Args:
            model_path (str): Path to the HPSv2 model checkpoint.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        try:
            self.model, _, self.preprocess_function = create_model_and_transforms(
                "ViT-H-14",
                "laion2B-s32B-b79K",
                precision="amp",
                device=self.device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False
            )

            checkpoint = torch.load(self.model_path)
            if "state_dict" not in checkpoint:
                raise ModelLoadingError("Checkpoint does not contain 'state_dict'.")

            self.model.load_state_dict(checkpoint["state_dict"])
            self.tokenizer = get_tokenizer("ViT-H-14")
            self.model.eval()

        except FileNotFoundError as e:
            raise ModelLoadingError(f"Model checkpoint not found at '{self.model_path}'.") from e
        except Exception as e:
            raise ModelLoadingError(f"Error loading model: {e}") from e

    def inference(self, inputs: torch.Tensor, captions: Union[List[str], torch.Tensor]) -> List[float]:
        """
        Runs inference on a batch of images and corresponding captions.
        Returns a batch of reward scores.
        """
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("Expected 'inputs' to be a list of PIL.Image objects.")
        if not (isinstance(captions, torch.Tensor) or (isinstance(captions, list) and all(isinstance(c, str) for c in captions))):
            raise TypeError("Expected 'captions' to be either a torch.Tensor or a list of strings.")
        if len(inputs) != len(captions):
            raise ValueError("Number of 'inputs' and 'captions' must match.")
        
        try:
            with torch.no_grad():
                if not isinstance(captions, torch.Tensor):
                    text_tokens = self.tokenizer(captions).to(self.device)
                else:
                    text_tokens = captions.to(self.device)
                inputs = inputs.to(self.device)
                    
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs, text_tokens)
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    similarity_scores = (image_features @ text_features.T).diag() * 100   
                return similarity_scores.tolist()
            
        except Exception as e:
            raise InferenceError(f"Inference failed: {e}") from e


    def inference_with_grad(self, inputs: torch.Tensor, captions: List[str]) -> List[float]:
        """
        Runs inference on a batch of images and corresponding captions.
        Returns a batch of reward scores.
        """
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("Expected 'inputs' to be a list of PIL.Image objects.")
        if not isinstance(captions, list) or not all(isinstance(c, str) for c in captions):
            raise TypeError("Expected 'captions' to be a list of strings.")
        if len(inputs) != len(captions):
            raise ValueError("Number of 'inputs' and 'captions' must match.")
        
        try:
            text_tokens = self.tokenizer(captions).to(self.device)
            inputs = inputs.to(self.device)

            with torch.cuda.amp.autocast():
                outputs = self.model(inputs, text_tokens)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                return (image_features @ text_features.T).diag() * 100
            
        except Exception as e:
            raise InferenceError(f"Inference failed: {e}") from e