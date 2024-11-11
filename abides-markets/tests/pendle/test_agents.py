import numpy as np

from abides_core.kernel import Kernel
from abides_markets.agents import TradingAgent, LiquidatorAgent
from abides_core.utils import str_to_ns
from abides_markets.agents.utils import tick_to_rate

from abides_markets.rate_oracle import ConstantOracle


def test_maintainance_margin():
    agent = TradingAgent(id=0)
    agent.rate_oracle = ConstantOracle()

    agent.position = {"COLLATERAL": 100,
                      "SIZE": 0,
                      "FIXRATE": 0}
    
    agent.mkt_open = 0
    agent.mkt_close = 365*str_to_ns("1d")
    agent.current_time = 0

    assert agent.maintainance_margin() == 0
    assert agent.maintainance_margin(10) == 0.3
    assert agent.maintainance_margin(20) == 0.6
    assert agent.maintainance_margin(60) == 3
    assert round(agent.maintainance_margin(110), 1) == 6.4

class FakeOrderBook:
    def __init__(self):
        pass

    def get_twap(self):
        return 1000
    
class FakeOracle:
    def __init__(self):
        pass

def test_mark_to_market():
    agent = TradingAgent(id=0)

    kernel = Kernel([agent], 
                    swap_interval = str_to_ns("8h"),
                    )
    kernel.book = FakeOrderBook()

    agent.kernel = kernel

    agent.position = {"COLLATERAL": 100,
                      "SIZE": 100,
                      "FIXRATE": 0.20}
    
    agent.mkt_open = 0
    agent.mkt_close = 365*str_to_ns("1d")
    agent.current_time = 0

    assert agent.mark_to_market(market_tick=1500, log=False) == 100 + 100 * (tick_to_rate(1500) - 0.20)
    assert agent.mark_to_market(log=False) == 100 + 100 * (tick_to_rate(1000) - 0.20)

def test_liquidation_status():
    agent = TradingAgent(id=0)

    kernel = Kernel([agent], 
                    swap_interval = str_to_ns("8h"),
                    )
    kernel.book = FakeOrderBook()

    agent.kernel = kernel

    agent.position = {"COLLATERAL": 20,
                      "SIZE": 100,
                      "FIXRATE": 0.20}
    
    agent.mkt_open = 0
    agent.mkt_close = 365*str_to_ns("1d")
    agent.current_time = 0

    assert round(agent.mark_to_market(log=False) - (20 + 100 * (tick_to_rate(1000) - 0.20)), 4) == 0
    assert round(agent.maintainance_margin(100) - 5.4, 4) == 0

    assert round(agent.mRatio() - 5.4/(20 + 100*(tick_to_rate(1000) - 0.20)), 4) == 0
    assert agent.is_healthy()

    agent.position = {"COLLATERAL": 14,
                      "SIZE": 100,
                      "FIXRATE": 0.20}
    
    assert not agent.is_healthy()

def test_liquidator():
    liquidator = LiquidatorAgent(id=0)
    dummy_agent = TradingAgent(id=1)

    kernel = Kernel([liquidator, dummy_agent], 
                    swap_interval = str_to_ns("8h"),
                    )
    kernel.book = FakeOrderBook()

    dummy_agent.kernel = kernel
    liquidator.kernel = kernel

    dummy_agent.position = {"COLLATERAL": 14,
                            "SIZE": 100,
                            "FIXRATE": 0.20}
    dummy_agent.mkt_open = 0
    dummy_agent.mkt_close = 365*str_to_ns("1d")
    dummy_agent.current_time = 0
    
    liquidator.known_bids[liquidator.symbol] = [
        [1000, 10],
        [950, 10]
    ]

    liquidator.check_liquidate(dummy_agent, sell=False)
    assert dummy_agent.position["SIZE"] == 90
    assert dummy_agent.position["COLLATERAL"] > 0
    assert dummy_agent.position["COLLATERAL"] < 14



    



