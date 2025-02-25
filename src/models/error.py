class ModelLoadingError(Exception):
    """Exception raised when there is an error loading the model."""
    pass

class InferenceError(Exception):
    """Exception raised when an error occurs during inference."""
    pass
