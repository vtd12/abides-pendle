import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from abides_markets.rate_oracle import ConstantOracle
from abides_markets.rate_oracle import BTCOracle
from abides_markets.driving_oracle import ManualOracle
from abides_markets.agents import ValueAgent

from abides_core import Kernel
from abides_core.utils import datetime_str_to_ns, str_to_ns


def test_const_oracle():
    const_oracle = ConstantOracle(0, 0, 10)

    a = const_oracle.get_floating_rate(12345)
    b = const_oracle.get_floating_rate(12345)

    assert a == b

def test_BTC_oracle():
    oracle = BTCOracle()
    assert round(oracle.get_floating_rate(1_673_222_400_003_000_001) - 5.796000e-05, 10) == 0
    assert oracle.get_floating_rate(1_675_238_400_000_000_000) == 0.0001
    

    assert True

def test_driving_oracle():
    MKT_OPEN = int(pd.to_datetime("20230101").to_datetime64())
    MKT_CLOSE = int(pd.to_datetime("20230201").to_datetime64())
    ticker = "PEN"

    # driving oracle
    symbols = {
        ticker: {
            "r_bar": 1000,
            "kappa": 0.001,
            "sigma_s": 100
        }
    }

    driving_oracle = ManualOracle(MKT_OPEN, MKT_CLOSE, symbols)
    oracle_value = driving_oracle.r[ticker]

    assert len(oracle_value) == 31*24*60+1

    plt.plot(oracle_value)
    plt.savefig('log/oracle_value_plot.png')

    current_time = datetime_str_to_ns("20230106")

    assert driving_oracle.observe_price(ticker, current_time, np.random.RandomState(0)) == 1321  # Check if observe works

    value_agent = ValueAgent(id=0)
    value_agent.driving_oracle = driving_oracle
    value_agent.rate_oracle = ConstantOracle()
    kernel = Kernel([value_agent], 
                    swap_interval = str_to_ns("8h"),
                    )
    value_agent.kernel = kernel

    value_agent.current_time = current_time

    value_agent.updateEstimates()

    assert value_agent.r_t

    