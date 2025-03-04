from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Dict, List

from datasets.error import DatasetFormatError, DatasetLoadingError

class BasePromptDataset(Dataset, ABC):
    def __init__(self):
        try:
            self.data = self.load_dataset()
        except Exception as e:
            raise DatasetLoadingError(f"Failed to load dataset: {e}")

        if not isinstance(self.data, dict):
            raise DatasetFormatError(f"Expected 'load_dataset()' to return a dictionary, got '{type(self.data)}'.")

        for key, prompts in self.data.items():
            if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
                raise DatasetFormatError(f"Expected a list of strings for category '{key}', but got '{type(prompts)}'")

        # Precompute samples with round-robin ordering
        self.samples = self._create_round_robin_samples()

    @abstractmethod
    def load_dataset(self) -> Dict[str, List[str]]:
        """To be implemented by subclasses."""
        pass

    def _create_round_robin_samples(self) -> List[Dict[str, str]]:
        """Ensure fair round-robin interleaving of prompts from all categories."""
        samples = []
        categories = list(self.data.keys())
        category_prompts = [self.data[cat] for cat in categories]

        if not categories or all(len(prompts) == 0 for prompts in category_prompts):
            raise DatasetFormatError("Dataset is empty or contains only empty categories.")

        max_length = max(len(prompts) for prompts in category_prompts)

        # Round-robin interleaving
        for i in range(max_length):
            for cat_idx, category in enumerate(categories):
                prompts = category_prompts[cat_idx]
                if len(prompts) > 0:
                    prompt = prompts[i % len(prompts)]  # Cycle back for shorter lists
                    samples.append({"category": category, "prompt": prompt})

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def num_categories(self) -> int:
        """Returns the number of unique categories in the dataset."""
        return len(self.data)