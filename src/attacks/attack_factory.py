from models.base_model import BaseModel
from attacks.base_attack import BaseAttack
from attacks.white_box.fgsm_reward_model import FGSMRewardModel
from attacks.white_box.pgd_reward_model import PGDRewardModel
from attacks.black_box.gn_reward_model import GNRewardModel
from attacks.black_box.spsa_reward_model import SPSARewardModel

class AttackFactory:
    @staticmethod
    def create_dataset(
        attack_type: str,
        model: BaseModel,
        **kwargs,
    ) -> BaseAttack:
        if attack_type == "gn":
            return GNRewardModel(model)
        elif attack_type == "fgsm":
            return FGSMRewardModel(model, **kwargs)
        elif attack_type == "pgd":
            return PGDRewardModel(model, **kwargs)
        elif attack_type == "spsa":
            return SPSARewardModel(model, **kwargs)
        else:
            raise ValueError("Unsupported attack type.")