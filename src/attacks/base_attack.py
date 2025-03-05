from collections import OrderedDict

import torch
from torchattacks.attack import Attack, wrapper_method


class BaseAttack(Attack):
    """
    Small modifications to the torchattack's Attack class
    to work with reward models
    """
    def __init__(self, name, model):
        """
        Initializes internal attack state.

        Arguments:
            name (str): name of attack.
            model (BaseModel): model to attack.
        """

        self.attack = name
        self._attacks = OrderedDict()

        self.set_model(model)
        ################################################
        # MODIFICATION
        # Set device using torch.cuda instead of
        # model.parameters().device
        ################################################        
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ################################################
        except Exception:
            self.device = None
            print("Failed to set device automatically, please try set_device() manual.")

        # Controls attack mode.
        self.attack_mode = "default"
        self.supported_mode = ["default"]
        self.targeted = False
        self._target_map_function = None

        # Controls when normalization is used.
        self.normalization_used = None
        self._normalization_applied = None
        if self.model.__class__.__name__ == "RobModel":
            self._set_rmodel_normalization_used(model)

        # Controls model mode during attack.
        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False        

    @wrapper_method
    def _change_model_mode(self, given_training):
        ################################################
        # MODIFICATION
        # do not iterate over model parameters
        # as we use pipelines for inference
        ################################################  
        pass
        # if self._model_training:
        #     self.model.train()
        #     for _, m in self.model.named_modules():
        #         if not self._batchnorm_training:
        #             if "BatchNorm" in m.__class__.__name__:
        #                 m = m.eval()
        #         if not self._dropout_training:
        #             if "Dropout" in m.__class__.__name__:
        #                 m = m.eval()
        # else:
        #     self.model.eval()

    @wrapper_method
    def _recover_model_mode(self, given_training):
        ################################################
        # MODIFICATION
        # do not execute model.train()
        # as we use pipelines for inference
        ################################################ 
        if given_training:
            pass
            # self.model.train()