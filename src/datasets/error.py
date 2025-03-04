class DatasetFormatError(Exception):
    """Raised when the dataset format is incorrect."""
    pass

class DatasetLoadingError(Exception):
    """Raised when the dataset fails to load properly."""
    pass