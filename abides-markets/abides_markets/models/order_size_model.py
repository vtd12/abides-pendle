import json

import numpy as np
from pomegranate import GeneralMixtureModel


order_size = {
    "class": "GeneralMixtureModel",
    "distributions": [
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [120_000, 1_500],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [600_000, 1_500],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [3_000_000, 1_500],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [15_000_000, 1_500],
            "frozen": True,
        },
    ],
    "weights": [
        0.8,
        0.15, 
        0.04,
        0.01
    ],
}


class OrderSizeModel:
    def __init__(self) -> None:
        self.model = GeneralMixtureModel.from_json(json.dumps(order_size))

    def sample(self, random_state: np.random.RandomState) -> float:
        return round(self.model.sample(random_state=random_state))
    
class NormalOrderSizeModel:
    def __init__(self, mean, sd) -> None:
        self.mean = mean
        self.sd = sd

    def sample(self, random_state: np.random.RandomState) -> float:
        return round(np.random.normal(self.mean, self.sd))