import hpsv2
from typing import Dict, List

from datasets.base_prompt_dataset import BasePromptDataset

class HPSV2PromptDataset(BasePromptDataset):
    def load_dataset(self) -> Dict[str, List[str]]:
        all_prompts = hpsv2.benchmark_prompts("all")
        return dict(all_prompts.items())