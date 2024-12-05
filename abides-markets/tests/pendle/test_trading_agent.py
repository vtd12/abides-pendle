import pytest
import logging
import numpy as np
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from abides_markets.agents import TradingAgent, ExchangeAgent
from abides_markets.messages.market import MarketHoursMsg
from abides_core.kernel import Kernel

logger = logging.getLogger(__name__)

class FakeOrderBook:
    def __init__(self):
        self.last_twap = -3029.555281  # Example value

class FakeRateOracle:
    def __init__(self):
        pass
    
    def get_floating_rate(self, current_time: NanosecondTime) -> float:
        return -0.0003231358432291831  # Example value

def test_trading_agent_calculate():
    logger.debug("Starting test_trading_agent_calculate")

    mkt_open = 0
    mkt_close = 365 * str_to_ns("1d")  # 1 year in nanoseconds
    exchange_agent = ExchangeAgent(
        id=1,
        mkt_open=mkt_open,
        mkt_close=mkt_close,
        symbols=["PEN"],
        name="FakeExchange",
        type="ExchangeAgent",
        random_state=np.random.RandomState(seed=42),
        log_orders=False,
        use_metric_tracker=False  
    )

    exchange_agent.order_books["PEN"] = FakeOrderBook()

    unhealthy_agent = TradingAgent(id=0, name="UnhealthyAgent")

    swap_interval = str_to_ns("8h")
    kernel = Kernel(
        agents=[exchange_agent, unhealthy_agent], 
        swap_interval=swap_interval,
        custom_properties={"rate_oracle": FakeRateOracle()}
    )
    
    unhealthy_agent.kernel = kernel
    exchange_agent.kernel = kernel

    unhealthy_agent.kernel_starting(start_time=0)

    market_hours_msg = MarketHoursMsg(
        mkt_open=exchange_agent.mkt_open,
        mkt_close=exchange_agent.mkt_close
    )
    unhealthy_agent.receive_message(
        current_time=0,
        sender_id=exchange_agent.id,
        message=market_hours_msg
    )

    unhealthy_agent.position = {"COLLATERAL": 50, "SIZE": 100, "FIXRATE": 0.1095}

    kernel.book = exchange_agent.order_books["PEN"]

    unhealthy_agent.position_updated()

    mark_to_market = unhealthy_agent.mark_to_market(unhealthy_agent.position)
    expected_mtm = 3.667
    assert round(mark_to_market, 3) == expected_mtm, f"Expected mark to market to be {expected_mtm}, got {mark_to_market}"
    logger.debug(f"Agent's mark to market: {mark_to_market}")
    
    maintenance_margin = unhealthy_agent.maintainance_margin()
    expected_mm = 3.8
    assert round(maintenance_margin , 3) == expected_mm, f"Expected maintenance margin to be {expected_mm}, got {maintenance_margin}"
    logger.debug(f"Agent's maintenance margin: {maintenance_margin}")
    
    m_ratio = unhealthy_agent.mRatio(unhealthy_agent.position)
    expected_mratio = 0.965
    assert round(m_ratio, 3) == expected_mratio, f"Expected mRatio to be {expected_mratio}, got {m_ratio}"
    logger.debug(f"Agent's mRatio: {m_ratio}")
    
    health_status = unhealthy_agent.is_healthy()
    expected_health = False
    assert health_status == expected_health, f"Expected health status to be {expected_health}, got {health_status}"
    logger.debug(f"Agent's health status: {health_status}")
