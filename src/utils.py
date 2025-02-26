import gc
import torch

def clear_cuda_memory_and_force_gc(force: bool = False):
    """
    Clears the CUDA memory cache and forces garbage collection if the allocated memory
    exceeds a certain threshold or if explicitly forced.

    Args:
        force (bool): If True, CUDA cache will be cleared and garbage collection
                      will be forced regardless of the memory threshold.
    """

    memory_allocated = torch.cuda.max_memory_reserved()
    memory_total = torch.cuda.get_device_properties("cuda").total_memory

    memory_threshold = memory_total * 0.7
    if memory_allocated > memory_threshold or force:
        torch.cuda.empty_cache()
        gc.collect()
