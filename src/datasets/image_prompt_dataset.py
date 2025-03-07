from typing import List, Tuple

import PIL
from torch.utils.data import Dataset

from datasets.error import DatasetFormatError
class ImagePromptDataset(Dataset):
    def __init__(
            self, 
            image_list: List[PIL.Image], prompt_list: List[Tuple[str, str]], 
            image_transform_function: callable, text_tokenizer_function: callable = None
        ):
        """
        Args:
            image_list (List[PIL.Image]): List of PIL images.
            prompt_list (List[Tuple[str, str]]): List of (category, prompt) tuples.
            image_transform_function (callable): Function to transform PIL images.
            text_tokenizer_function (callable): Function to tokenize text prompts.
        """
        if len(image_list) == 0 or len(prompt_list) == 0:
            raise DatasetFormatError("Both image_list and prompt_list must be non-empty.")
        if len(image_list) != len(prompt_list):
            raise DatasetFormatError("Images and prompts must have the same length.")

        self.images = image_list
        self.prompts = prompt_list  # List of (category, prompt)
        self.image_transform_function = image_transform_function
        self.text_tokenizer_function = text_tokenizer_function

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.image_transform_function(self.images[idx])
        _, prompt = self.prompts[idx]
        if self.text_tokenizer_function is None:
            tokens = prompt
        else:
            tokens = self.text_tokenizer_function(prompt)

        return image, tokens
