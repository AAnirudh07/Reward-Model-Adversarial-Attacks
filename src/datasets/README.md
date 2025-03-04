### Dataset Files Inheritance Tree

```
BasePromptDataset (base_prompt_dataset.py)
├── HPSV2PromptDataset (hpsv2_benchmark_dataset.py)
└── DrawBenchPromptDataset (drawbench_dataset.py)

ImagePromptDataset (image_prompt_dataset.py)
```

- `RoundRobinSampler` (round_robin_sampler.py) -- Custom sampler for fair round-robin interleaving of samples from all categories. It precomputes sample indices for balanced sampling.