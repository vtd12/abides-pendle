import numpy as np

from abides_markets.rate_oracle import ConstantOracle
from abides_markets.rate_oracle import BTCOracle


def test_const_oracle():
    const_oracle = ConstantOracle(0, 0, 10)

    a = const_oracle.get_floating_rate(12345)
    b = const_oracle.get_floating_rate(12345)

    assert a == b

def test_BTC_oracle():
    oracle = BTCOracle()
    assert oracle.get_floating_rate(1675238400000) == 0.0001
    assert round(oracle.get_floating_rate(1706601600001) - 0.000095, 6) == 0
    

    assert True
