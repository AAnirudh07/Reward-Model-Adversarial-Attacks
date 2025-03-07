import clip
import torch
from typing import List, Union
from models.base_model import BaseModel
from models.error import ModelLoadingError, InferenceError

class HPSv1Model(BaseModel):
    def __init__(self, model_path: str):
        """
        Args:
            model_path (str): Path to the HPSv1 model checkpoint.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        try:
            self.model, self.preprocess_function = clip.load("ViT-L/14", device=self.device)
            checkpoint = torch.load(self.model_path)

            if "state_dict" not in checkpoint:
                raise ModelLoadingError("Checkpoint does not contain 'state_dict'.")
            
            self.model.load_state_dict(checkpoint["state_dict"])
            self.tokenizer = clip.tokenize
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
            raise TypeError("Expected 'inputs' to be of type torch.Tensor (i.e. images).")
        if not isinstance(captions, list) or not all(isinstance(c, str) for c in captions):
            raise TypeError("Expected 'captions' to be a list of strings.")
        if inputs.shape[0] != len(captions):
            raise ValueError("Number of 'inputs' and 'captions' must match.")
        
        try:
            with torch.no_grad():
                image_features = self.model.encode_image(inputs.to(self.device))

                if not isinstance(captions, torch.Tensor):
                    text_tokens = self.tokenizer(captions).to(self.device)
                else:
                    text_tokens = captions.to(self.device)
                text_features = self.model.encode_text(text_tokens)

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Convert cosine similarity scores to percentages as in the original paper
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
            raise TypeError("Expected 'inputs' to be of type torch.Tensor (i.e. images).")
        if not isinstance(captions, list) or not all(isinstance(c, str) for c in captions):
            raise TypeError("Expected 'captions' to be a list of strings.")
        if inputs.shape[0] != len(captions):
            raise ValueError("Number of 'inputs' and 'captions' must match.")
        
        try:
            text_tokens = clip.tokenize(captions).to(self.device)
            image_features, text_features = self.model(inputs, text_tokens)
            return (image_features @ text_features.T).diag() * 100
        except Exception as e:
            raise InferenceError(f"Inference failed: {e}") from e