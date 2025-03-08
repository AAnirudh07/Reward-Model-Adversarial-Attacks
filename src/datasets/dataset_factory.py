from typing import Union

from datasets.base_prompt_dataset import BasePromptDataset
from datasets.drawbench_dataset import DrawBenchPromptDataset
from datasets.hpsv2_benchmark_dataset import HPSV2PromptDataset
from datasets.image_prompt_dataset import ImagePromptDataset

class DatasetFactory:
    @staticmethod
    def create_dataset(
        dataset_type: str,
        **kwargs,
    ) -> Union[BasePromptDataset, ImagePromptDataset]:
        
        if dataset_type == "drawbench":
            return DrawBenchPromptDataset()
        elif dataset_type == "hps":
            return HPSV2PromptDataset()
        elif dataset_type == "imageandprompt":
            return ImagePromptDataset(**kwargs)
        else:
            raise ValueError(f"Unknown dataset type: '{dataset_type}'.")