import json

import numpy as np
from pomegranate import GeneralMixtureModel


order_size = {
    "class": "GeneralMixtureModel",
    "distributions": [
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [1_000_000, 1_500],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [2_000_000, 1_500],
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
            "parameters": [4_000_000, 1_500],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [5_000_000, 1_500],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [6_000_000, 1_500],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [7_000_000, 1_500],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [8_000_000, 1_500],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [9_000_000, 1_500],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [10_000_000, 1_500],
            "frozen": True,
        },
    ],
    "weights": [
        0.9,
        0.06,
        0.004,
        0.0329,
        0.001,
        0.0006,
        0.0004,
        0.0005,
        0.0003,
        0.0003,
    ],
}


class OrderSizeModel:
    def __init__(self) -> None:
        self.model = GeneralMixtureModel.from_json(json.dumps(order_size))

    def sample(self, random_state: np.random.RandomState) -> float:
        return round(self.model.sample(random_state=random_state))
