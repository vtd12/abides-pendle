import pytest
import logging
import queue
import numpy as np
from datetime import datetime
import logging
from abides_core import Message, NanosecondTime
from typing import Any, Dict, List, Optional, Tuple, Type
from abides_core.agent import Agent
from abides_core.message import Message, MessageBatch, WakeupMsg, SwapMsg, UpdateRateMsg
from abides_core.utils import str_to_ns, merge_swap, fmt_ts
from abides_markets.agents import TradingAgent, LiquidatorAgent, ExchangeAgent
from abides_markets.orders import Side, MarketOrder
from abides_markets.messages.marketdata import MarketDataMsg, L2SubReqMsg
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.messages.order import MarketOrderMsg
from abides_markets.messages.market import MarketClosePriceRequestMsg
from abides_core.kernel import Kernel
FakeKernel = Kernel
logger = logging.getLogger(__name__)

class FakeOrderBook:
    def __init__(self):
        self.last_twap = -3029.555281
    
    def get_twap(self):
        return self.last_twap
    
    def set_wakeup(self, agent_id: int, requested_time: NanosecondTime) -> None:
        pass
class FakeOracle:
    def __init__(self):
        pass

class FakeRateOracle:
    def __init__(self):
        pass
    
    def get_floating_rate(self, current_time: NanosecondTime) -> float:
        return -0.0003231358432291831

def test_trading_agent_calculate():
    logger.debug("Starting test_trading_agent_calculate_liquidation")


    logger.debug("Initializing ExchangeAgent")
    exchange_agent = ExchangeAgent(
        id=1,
        mkt_open=0,
        mkt_close=365 * str_to_ns("1d"),
        symbols=["PEN"],
        name="FakeExchange",
        type="ExchangeAgent",
        random_state=np.random.RandomState(seed=42),
        log_orders=False,
        use_metric_tracker=False  
    )


    logger.debug("Setting FakeOrderBook for ExchangeAgent")
    exchange_agent.order_books["PEN"] = FakeOrderBook()


    logger.debug("Initializing TradingAgent")
    unhealthy_agent = TradingAgent(id=0, name="UnhealthyAgent")

    logger.debug("Creating and configuring Kernel")
    kernel = Kernel(
        agents=[exchange_agent, unhealthy_agent], 
        swap_interval=str_to_ns("8h"),
        custom_properties={"rate_oracle": FakeRateOracle()}
    )
    
    logger.debug(f"Kernel.book type: {type(kernel.book)}, last_twap: {getattr(kernel.book, 'last_twap', 'No last_twap')}")
    assert isinstance(kernel.book, FakeOrderBook), "Kernel.book is not FakeOrderBook"
    
    logger.debug("Assigning kernel to agents")
    unhealthy_agent.kernel = kernel
    exchange_agent.kernel = kernel


    logger.debug("Setting up unhealthy TradingAgent")
    unhealthy_agent.position = {"COLLATERAL": 50, "SIZE": 100, "FIXRATE": 0.1095}
    unhealthy_agent.mkt_open = 0
    unhealthy_agent.mkt_close = 365 * str_to_ns("1d")
    unhealthy_agent.current_time = 0
    
    logger.debug("Calculating initial values for unhealthy agent")
    maintenance_margin = unhealthy_agent.maintainance_margin(unhealthy_agent.position["SIZE"])
    assert round(maintenance_margin , 3) == 3.8, f"Expected maintenance margin to be 3.8, got {maintenance_margin}"
    logger.debug(f"Agent's maintenance margin: {maintenance_margin}")
    
    mark_to_market = unhealthy_agent.mark_to_market(unhealthy_agent.position)
    assert round(mark_to_market, 3) == 3.667, f"Expected mark to market to be 0.0, got {mark_to_market}"
    logger.debug(f"Agent's mark to market: {mark_to_market}")

    m_ratio = unhealthy_agent.mRatio(unhealthy_agent.position)
    assert round(m_ratio, 3) == 1.036, f"Expected mRatio to be 1.036, got {m_ratio}"
    logger.debug(f"Agent's mRatio: {m_ratio}")

    health_status = unhealthy_agent.is_healthy()
    assert not health_status, f"Expected health status to be False, got {health_status}"
    logger.debug(f"Agent's health status: {health_status}")