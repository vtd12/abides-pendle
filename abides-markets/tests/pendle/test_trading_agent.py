from matplotlib.pylab import f
import numpy as np

from abides_core.kernel import Kernel
from abides_markets.agents import TradingAgent
from abides_core.utils import str_to_ns
from abides_markets.rate_oracle import ConstantOracle
from abides_markets.agents.utils import tick_to_rate, rate_to_tick

class FakeOrderBook:
    """A fake order book to provide a constant TWAP value."""

    def __init__(self, twap_value=1000):
        self.twap_value = twap_value

    def get_twap(self):
        return self.twap_value  # Mocked market price tick value



def test_maintainance_margin():
    agent = TradingAgent(id=0)
    agent.pen_oracle = ConstantOracle()

    # Initialize agent's position
    agent.position = {"COLLATERAL": 100,
                      "SIZE": 0,
                      "FIXRATE": 0}

    # Set market open and close times
    agent.mkt_open = 0
    agent.mkt_close = 365 * str_to_ns("1d")  # 365 days in nanoseconds
    agent.current_time = 0

    # Test different position sizes
    assert agent.maintainance_margin() == 0
    assert agent.maintainance_margin(10) == 0.3
    assert agent.maintainance_margin(20) == 0.6
    assert agent.maintainance_margin(60) == 3.0
    assert round(agent.maintainance_margin(110), 1) == 6.4

# TODO: Check self.kernel.swap_interval
def test_mark_to_market():
    agent = TradingAgent(id=0)

    # Initialize kernel and assign to agent
    kernel = Kernel([agent],
                    swap_interval=str_to_ns("8h"),
                    )
    kernel.book = FakeOrderBook()
    kernel.rate_normalizer = 1  # Assume normalization factor is 1

    agent.kernel = kernel

    # Initialize agent's position
    agent.position = {"COLLATERAL": 100,
                      "SIZE": 100,
                      "FIXRATE": 0.20}

    # Set market open and close times
    agent.mkt_open = 0
    agent.mkt_close = 365 * str_to_ns("1d")  # 365 days in nanoseconds
    agent.current_time = 0

    # Calculate the number of payments remaining
    n_payment = int((agent.mkt_close - agent.current_time) // agent.kernel.swap_interval)
    # Test with a given market tick value (1500)
    market_tick = 1500
    market_rate = tick_to_rate(market_tick)
    expected_value = agent.position["COLLATERAL"] + agent.position["SIZE"] * (market_rate - agent.position["FIXRATE"]) * n_payment
    result = agent.mark_to_market(market_tick=market_tick, log=False)
    assert round(result, 6) == round(expected_value, 6)

    # Test using the default market price from the fake order book (1000)
    market_tick = agent.kernel.book.get_twap()
    market_rate = tick_to_rate(market_tick)
    expected_value = agent.position["COLLATERAL"] + agent.position["SIZE"] * (market_rate - agent.position["FIXRATE"]) * n_payment
    result = agent.mark_to_market(log=False)
    assert round(result, 6) == round(expected_value, 6)


def test_liquidation_status():
    agent = TradingAgent(id=0)

    kernel = Kernel([agent],
                    swap_interval=str_to_ns("8h"),
                    )
    kernel.book = FakeOrderBook()
    kernel.rate_normalizer = 1

    agent.kernel = kernel

    # Initialize agent's position
    agent.position = {"COLLATERAL": 20,
                      "SIZE": 100,
                      "FIXRATE": 0.20}

    agent.mkt_open = 0
    agent.mkt_close = 365 * str_to_ns("1d")
    agent.current_time = 0

    # Calculate the number of payments remaining
    n_payment = int((agent.mkt_close - agent.current_time) // agent.kernel.swap_interval)

    # Calculate expected mark-to-market value
    market_tick = agent.kernel.book.get_twap()
    market_rate = tick_to_rate(market_tick)
    expected_mtm = agent.position["COLLATERAL"] + agent.position["SIZE"] * (market_rate - agent.position["FIXRATE"]) * n_payment
    assert round(agent.mark_to_market(log=False), 4) == round(expected_mtm, 4)

    # Test maintainance_margin function
    assert round(agent.maintainance_margin(100), 4) == 5.4

    # Test mRatio function
    expected_mratio = 5.4 / expected_mtm
    assert round(agent.mRatio(), 4) == round(expected_mratio, 4)

    # Test is_healthy function
    assert agent.is_healthy()

    # Modify COLLATERAL to make the agent unhealthy
    agent.position["COLLATERAL"] = 14
    assert not agent.is_healthy()


def test_merge_swap():
    agent = TradingAgent(id=0)
    agent.position = {"COLLATERAL": 1000,
                      "SIZE": 100,
                      "FIXRATE": 0.05}

    # Merge a new swap into the position
    p_merge_pa = agent.merge_swap(50, 0.06)
    expected_size = 150
    expected_rate = (100 * 0.05 + 50 * 0.06) / 150
    assert agent.position["SIZE"] == expected_size
    assert round(agent.position["FIXRATE"], 6) == round(expected_rate, 6)
    assert agent.position["COLLATERAL"] == 1000 + p_merge_pa

    # Merge a negative swap (reduce position size)
    p_merge_pa = agent.merge_swap(-30, 0.055)
    expected_size = 120
    expected_rate = (150 * expected_rate - 30 * 0.055) / 120
    assert agent.position["SIZE"] == expected_size
    assert round(agent.position["FIXRATE"], 6) == round(expected_rate, 6)
    assert agent.position["COLLATERAL"] == 1000 + p_merge_pa

# TODO: FAILED test_agents.py::test_R2 - ValueError: n_payment in R2 calculation is zero.
def test_R2():
    agent = TradingAgent(id=0)

    agent.mkt_open = 0
    agent.mkt_close = 365 * str_to_ns("1d")
    agent.current_time = 0

    kernel = Kernel([agent],
                    swap_interval=str_to_ns("1d"),
                    )
    kernel.rate_normalizer = 1
    agent.kernel = kernel

    agent.position = {"COLLATERAL": 10,
                      "SIZE": 100,
                      "FIXRATE": 0.05}

    # Calculate the number of payments remaining
    n_payment = int((agent.mkt_close - agent.current_time) // agent.kernel.swap_interval)

    # Calculate sensitive rate and tick
    mm = agent.maintainance_margin(agent.position["SIZE"])
    sensitive_rate = (mm - agent.position["COLLATERAL"]) / (agent.kernel.rate_normalizer * agent.position["SIZE"] * n_payment) + agent.position["FIXRATE"]
    sensitive_tick = rate_to_tick(sensitive_rate)

    result = agent.R2()
    assert round(result, 6) == round(sensitive_tick, 6)

# TODO: FAILED test_agents.py::test_mRatio_and_is_healthy - AttributeError: 'Kernel' object has no attribute 'book'
def test_mRatio_and_is_healthy():
    agent = TradingAgent(id=0)

    kernel = Kernel([agent],
                    swap_interval=str_to_ns("1d"),
                    )
    kernel.rate_normalizer = 1
    agent.kernel = kernel

    agent.mkt_open = 0
    agent.mkt_close = 365 * str_to_ns("1d")
    agent.current_time = 0

    # Test with a healthy position
    agent.position = {"COLLATERAL": 50,
                      "SIZE": 100,
                      "FIXRATE": 0.05}
    m_ratio = agent.mRatio()
    assert m_ratio < 1
    assert agent.is_healthy()

    # Test with an unhealthy position
    agent.position = {"COLLATERAL": 1,
                      "SIZE": 100,
                      "FIXRATE": 0.05}
    m_ratio = agent.mRatio()
    assert m_ratio >= 1
    assert not agent.is_healthy()


def test_swap():
    agent = TradingAgent(id=0)

    kernel = Kernel([agent],
                    swap_interval=str_to_ns("1d"),
                    )
    kernel.rate_normalizer = 1
    agent.kernel = kernel

    agent.mkt_open = 0
    agent.mkt_close = 365 * str_to_ns("1d")
    agent.current_time = 0

    # Initialize agent's position
    agent.position = {"COLLATERAL": 1000,
                      "SIZE": 100,
                      "FIXRATE": 0.05}

    # Test swap method with floating rate of 0.06
    floating_rate = 0.06
    current_time = agent.current_time + str_to_ns("1d")  # Advance current time by one day
    agent.swap(current_time=current_time, floating_rate=floating_rate)

    # Calculate expected COLLATERAL
    expected_change = 100 * (0.06 - 0.05 * kernel.rate_normalizer)
    expected_collateral = 1000 + expected_change

    assert agent.position["COLLATERAL"] == expected_collateral
    assert agent.current_time == current_time

    # Perform another swap with floating rate of 0.055
    floating_rate = 0.055
    current_time += str_to_ns("1d")  # Time advances by another day
    agent.swap(current_time=current_time, floating_rate=floating_rate)

    # Update expected COLLATERAL
    expected_change += 100 * (0.055 - 0.05 * kernel.rate_normalizer)
    expected_collateral = 1000 + expected_change

    assert agent.position["COLLATERAL"] == expected_collateral
    assert agent.current_time == current_time
