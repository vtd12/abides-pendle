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
        return -0.0003231358432291831

def test_liquidator_calculate():
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
    liquidator.known_bids = {"PEN": [(-3029.555281, 5000)]}
    liquidator.known_asks = {"PEN": [(-3029.555281, 5000)]}
    liquidator.mkt_open = True
    liquidator.mkt_closed = False
    
    placed_liquidator_orders = []
    def mock_place_market_order(symbol, quantity, side):
        placed_liquidator_orders.append({'symbol': symbol, 'quantity': quantity, 'side': side})
    
    liquidator.place_market_order = mock_place_market_order
    
    unhealthy_agent.position = {"COLLATERAL": 50, "SIZE": 100, "FIXRATE": 0.1095}
    unhealthy_agent.mkt_open = 0
    unhealthy_agent.mkt_close = 365 * str_to_ns("1d")
    unhealthy_agent.current_time = 0
    unhealthy_agent.exchange_id = 0
    

    health_status = unhealthy_agent.is_healthy()
    assert not health_status
    
    liquidator.watch_list = [unhealthy_agent.id]
    
    current_time = 0
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
    
    with patch('abides_markets.agents.value_agent.tick_to_rate') as mock_tick_to_rate:
        mock_tick_to_rate.return_value = 0.07
        liquidator.check_liquidate(unhealthy_agent)
    
    
    assert unhealthy_agent.position["SIZE"] == 0
    assert round(unhealthy_agent.position["COLLATERAL"], 2) == 2.04
