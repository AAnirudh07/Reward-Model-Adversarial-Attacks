from typing import Dict, List

import pandas as pd

from datasets.base_prompt_dataset import BasePromptDataset

class DrawBenchPromptDataset(BasePromptDataset):
    def load_dataset(self) -> Dict[str, List[str]]:
        df = pd.read_csv("datasets/drawbench_data.csv")
        return df.groupby("Category")["Prompts"].apply(list).to_dict()