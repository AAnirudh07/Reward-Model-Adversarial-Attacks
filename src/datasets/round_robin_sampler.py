import torch
import random
from datasets.base_prompt_dataset import BasePromptDataset

class RoundRobinSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: BasePromptDataset):
        self.dataset = dataset
        self.indices = self._generate_indices()

    def _generate_indices(self):
        """
        Assume dataset.data has equal length lists per category.

        For each category, create a shuffled list of indices corresponding to that category's samples.
        Since BasePromptDataset precomputes samples in round-robin order, we need to map from category + position
        to the flat sample index.
        
        In our round-robin samples, the ordering is:
        index 0: category1, index 1: category2, ..., index N: category1
        
        Let K = number of categories,
        Then the sample index for category j at position i is: i*K + j.
        """
        categories = list(self.dataset.data.keys())
        num_per_category = len(next(iter(self.dataset.data.values())))
        K = len(categories)

        category_indices = {}
        for j, cat in enumerate(categories):
            indices = [i * K + j for i in range(num_per_category)]
            random.shuffle(indices)
            category_indices[cat] = indices

        ordered_indices = []
        for i in range(num_per_category):
            for cat in categories:
                ordered_indices.append(category_indices[cat][i])
        return ordered_indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)