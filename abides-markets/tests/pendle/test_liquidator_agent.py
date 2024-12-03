from py import log
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
    liquidator.mkt_close = 365 * str_to_ns("1d")
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
    
    logger.debug(f"maintainance_margin_of_new_position:{liquidator.maintainance_margin_of_new_position},liq_ict_fact: {liquidator.liq_ict_fact}, marginDelta: {liquidator.marginDelta}, liq_incentive:{liquidator.liq_incentive}, liq_val:{liquidator.liq_val}")
    assert round(unhealthy_agent.position["COLLATERAL"],2) == 3.53 , f"Expected 3.53, got {unhealthy_agent.position['COLLATERAL']}"
    logger.debug(f"p_unrealized:{liquidator.p_unrealized}")
    logger.debug(f"liq_incentive:{liquidator.liq_incentive}, liq_val:{liquidator.liq_val}")
    logger.debug(f"After liquidation, unhealthy_agent.position['COLLATERAL']: {unhealthy_agent.position['COLLATERAL']}")
    
    assert unhealthy_agent.position["SIZE"] == 0
    logger.debug(f"After liquidation, unhealthy_agent.position['SIZE']: {unhealthy_agent.position['SIZE']}")
    
    assert round(liquidator.position["COLLATERAL"], 2) == 100046.47
    logger.debug(f"After liquidation, liquidator.position['COLLATERAL']: {liquidator.position['COLLATERAL']}")

    assert round(liquidator.position["FIXRATE"], 2) == 0.11
    logger.debug(f"After liquidation, liquidator.position['FIXRATE']: {liquidator.position['FIXRATE']}")

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
    liquidator.mkt_close = 365 * str_to_ns("1d")
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
    
    logger.debug(f"p_unrealized:{liquidator.p_unrealized}")
    assert round(unhealthy_agent.position["COLLATERAL"],2) == 3.77 , f"Expected 3.77, got {unhealthy_agent.position['COLLATERAL']}" 
    assert unhealthy_agent.position["SIZE"] == 0
    assert round(liquidator.position["COLLATERAL"], 2) == 100046.23
    assert round(liquidator.position["FIXRATE"], 2) == -0.82
 
    # assert round(liquidator.position["SIZE"], 2) == 0
