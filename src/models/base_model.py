from abc import ABC, abstractmethod
from typing import Union, List, Optional
import torch

class BaseModel(ABC):
    @abstractmethod
    def load_model(self):
        """
        Load the open-weights model or make an API connection to the closed-source model.
        """
        pass

    @abstractmethod
    def inference(
        self, inputs: Union[List[str], torch.Tensor], captions: Optional[List[str]] = None
    ) -> Union[torch.Tensor, List[float]]:
        """
        Run inference on a batch of inputs with optional captions.
        
        Args:
            inputs (Union[List[str], torch.Tensor]): A batch of text prompts or a batch of images.
            captions (Optional[List[str]]): Optional text captions associated with the inputs for reward models.
        
        Returns:
            Union[torch.Tensor, List[float]]: A batch of model outputs or a list of reward scores.
        """
        pass
