from models.base_model import BaseModel
from models.hps_v1_model import HPSv1Model
from models.hps_v2_model import HPSv2Model
from models.stable_diffusion_model import StableDiffusionModel

class ModelFactory:
    @staticmethod
    def create_model(
        model_type: str, model_path: str,
        **kwargs,
    ) -> BaseModel:
        """
        Creates and returns an instance of a model subclass based on the model_type.

        Args:
            model_type (str): The type of model to create. Supported values are:
                - "hpsv1": For HPSv1 reward models.
                - "hpsv2": For HPSv2 reward models.
                - "sd": For stable diffusion text-to-image models.
            model_path (str): The path or repository ID of the model checkpoint.

        Returns:
            BaseModel: An instance of the requested model.

        Raises:
            ValueError: If an unsupported model_type is provided.
        """
        if model_type == "hpsv1":
            return HPSv1Model(model_path)
        elif model_type == "hpsv2":
            return HPSv2Model(model_path)
        elif model_type == "sd":
            return StableDiffusionModel(model_path, **kwargs)
        else:
            raise ValueError("Unsupported model type. Use 'sd' for stable diffusion models or 'hps' for HPS models.")