from typing import List, Tuple

import PIL
from torch.utils.data import Dataset

from datasets.error import DatasetFormatError
class ImageTextDataset(Dataset):
    def __init__(self, image_list: List[PIL.Image], prompt_list: List[Tuple[str, str]], transforms: callable):
        """
        Args:
            image_list (List[PIL.Image]): List of PIL images.
            prompt_list (List[Tuple[str, str]]): List of (category, prompt) tuples.
            transforms (callable): CLIP preprocessing functions.
        """
        if len(image_list) == 0 or len(prompt_list) == 0:
            raise DatasetFormatError("Both image_list and prompt_list must be non-empty.")
        if len(image_list) != len(prompt_list):
            raise DatasetFormatError("Images and prompts must have the same length.")

        self.images = image_list
        self.prompts = prompt_list  # List of (category, prompt)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transforms(self.images[idx])
        _, prompt = self.prompts[idx]

        return image, prompt
