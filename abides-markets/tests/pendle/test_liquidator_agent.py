import pytest
import logging
import numpy as np
from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns
from abides_markets.agents import TradingAgent, LiquidatorAgent, ExchangeAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from abides_core.kernel import Kernel
from abides_core.message import WakeupMsg
from unittest.mock import patch

logger = logging.getLogger(__name__)
FakeKernel = Kernel

class FakeOrderBook:
    def __init__(self):
        self.last_twap = -3029.555281
    
    def get_twap(self):
        return self.last_twap
    
    def set_wakeup(self, agent_id: int, requested_time: NanosecondTime) -> None:
        pass

class FakeRateOracle:
    def __init__(self):
        pass
    
    def get_floating_rate(self, current_time: NanosecondTime) -> float:
        return 0

def test_liquidator_calculate_into_bid_wall():
    logger.debug("Starting test_liquidator_calculate_liquidation")
    
    exchange_agent = ExchangeAgent(
        id=0,
        mkt_open=0,
        mkt_close=365 * str_to_ns("1d"),
        symbols=["PEN"],
        name="TestExchange",
        type="ExchangeAgent",
        random_state=np.random.RandomState(seed=42),
        log_orders=False,
        use_metric_tracker=False
    )

    exchange_agent.order_books["PEN"] = FakeOrderBook()

    liquidator = LiquidatorAgent(id=1, name="LiquidatorAgent")
    unhealthy_agent = TradingAgent(id=2, name="UnhealthyAgent")
    
    kernel = FakeKernel(agents=[exchange_agent, liquidator, unhealthy_agent], swap_interval=str_to_ns("8h"))
    
    liquidator.kernel = kernel
    unhealthy_agent.kernel = kernel
    exchange_agent.kernel = kernel
    
    liquidator.exchange_id = 0
    liquidator.symbol = "PEN"
    liquidator.known_bids = {"PEN": [(-3027.555281, 5000),(-3032.555281, 5000)]}
    liquidator.known_asks = {"PEN": [(-3027.555281, -5000),(-3026.555281, -5000)]}
    liquidator.mkt_open = 1
    liquidator.mkt_closed = False
    
    
    unhealthy_agent.position = {"COLLATERAL": 50, "SIZE": 100, "FIXRATE": 0.1095}
    unhealthy_agent.mkt_open = 1
    unhealthy_agent.mkt_close = 365 * str_to_ns("1d")
    unhealthy_agent.current_time = 2
    unhealthy_agent.exchange_id = 0
    

    health_status = unhealthy_agent.is_healthy()
    logger.debug(f"position: {unhealthy_agent.position}, health_status: {health_status}")
    assert not health_status
    liquidator.watch_list = [unhealthy_agent.id]
    
    current_time = 2
    liquidator.wakeup(current_time)
    
    message = QuerySpreadResponseMsg(
        symbol=liquidator.symbol,
        bids=liquidator.known_bids[liquidator.symbol],
        asks=liquidator.known_asks[liquidator.symbol],
        mkt_closed=False,
        depth=2,
        last_trade=None,
    )
    
    liquidator.receive_message(current_time, sender_id=0, message=message)
    
    liquidator.check_liquidate(unhealthy_agent)
    

    assert round(unhealthy_agent.position["COLLATERAL"],2) == 2.04 , f"Expected 5.25, got {unhealthy_agent.position['COLLATERAL']}" 
    assert unhealthy_agent.position["SIZE"] == 0
    assert round(liquidator.position["COLLATERAL"], 2) == 100047.96
    assert round(liquidator.position["FIXRATE"], 2) == 0.11
    logger.debug(f"self.sell_ask: {liquidator.sell_ask}, self.sell_bid: {liquidator.sell_bid},self.beforesell_position: {liquidator.beforesell_position}, self.after_sell_position: {liquidator.after_sell_position}")

    # assert round(liquidator.position["SIZE"], 2) == 0



def test_liquidator_calculate_into_ask_wall():
    logger.debug("Starting test_liquidator_calculate_liquidation")
    
    exchange_agent = ExchangeAgent(
        id=0,
        mkt_open=0,
        mkt_close=365 * str_to_ns("1d"),
        symbols=["PEN"],
        name="TestExchange",
        type="ExchangeAgent",
        random_state=np.random.RandomState(seed=42),
        log_orders=False,
        use_metric_tracker=False
    )

    exchange_agent.order_books["PEN"] = FakeOrderBook()

    liquidator = LiquidatorAgent(id=1, name="LiquidatorAgent")
    unhealthy_agent = TradingAgent(id=2, name="UnhealthyAgent")
    
    kernel = FakeKernel(agents=[exchange_agent, liquidator, unhealthy_agent], swap_interval=str_to_ns("8h"))
    
    liquidator.kernel = kernel
    unhealthy_agent.kernel = kernel
    exchange_agent.kernel = kernel
    
    liquidator.exchange_id = 0
    liquidator.symbol = "PEN"
    liquidator.known_bids = {"PEN": [(-3030.555281, 5000),(-3032.555281, 5000)]}
    liquidator.known_asks = {"PEN": [(-3030.555281, 5000),(-3026.555281, 5000)]}
    liquidator.mkt_open = 1
    liquidator.mkt_closed = False
    
    
    unhealthy_agent.position = {"COLLATERAL": 50, "SIZE": -100, "FIXRATE": -0.816}
    unhealthy_agent.mkt_open = 1
    unhealthy_agent.mkt_close = 365 * str_to_ns("1d")
    unhealthy_agent.current_time = 2
    unhealthy_agent.exchange_id = 0
    

    health_status = unhealthy_agent.is_healthy()
    logger.debug(f"position: {unhealthy_agent.position}, health_status: {health_status}")
    
    liquidator.watch_list = [unhealthy_agent.id]
    
    current_time = 2
    liquidator.wakeup(current_time)
    
    message = QuerySpreadResponseMsg(
        symbol=liquidator.symbol,
        bids=liquidator.known_bids[liquidator.symbol],
        asks=liquidator.known_asks[liquidator.symbol],
        mkt_closed=False,
        depth=2,
        last_trade=None,
    )
    
    liquidator.receive_message(current_time, sender_id=0, message=message)
    
    liquidator.check_liquidate(unhealthy_agent)
    

    assert round(unhealthy_agent.position["COLLATERAL"],2) == 3.58 , f"Expected 3.58, got {unhealthy_agent.position['COLLATERAL']}" 
    assert unhealthy_agent.position["SIZE"] == 0
    assert round(liquidator.position["COLLATERAL"], 2) == 100046.42
    assert round(liquidator.position["FIXRATE"], 2) == -0.82
    logger.debug(f"self.sell_ask: {liquidator.sell_ask}, self.sell_bid: {liquidator.sell_bid},self.beforesell_position: {liquidator.beforesell_position}, self.after_sell_position: {liquidator.after_sell_position}")

    # assert round(liquidator.position["SIZE"], 2) == 0
