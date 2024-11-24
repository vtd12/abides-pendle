import logging
import queue
import numpy as np
from abides_core.agent import Agent
from abides_core.rate_oracle import RateOracle
from abides_core import NanosecondTime
from abides_markets.agents.value_agent import ValueAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from datetime import datetime
from abides_core.utils import str_to_ns, merge_swap, fmt_ts
from abides_core import Message, NanosecondTime
from typing import Any, Dict, List, Optional, Tuple, Type
from abides_markets.agents import ExchangeAgent

from abides_core.kernel import Kernel
FakeKernel = Kernel
from abides_core.message import Message, MessageBatch, WakeupMsg, SwapMsg, UpdateRateMsg
logger = logging.getLogger(__name__)


class FakeOrderBook:
    def __init__(self):
        self.last_twap = -3.100998
    
    def get_twap(self):
        return self.last_twap
    
    def set_wakeup(self, agent_id: int, requested_time: NanosecondTime) -> None:
        pass
class FakeOracle:
    def __init__(self):
        pass

def test_value_agent():

    agent_id = 1
    symbol = "PEN"
    collateral = 100_000
    wake_up_freq = 600_000_000  
    r_bar = 0.10 
    coef = [0.05, 0.40]  

    exchange_agent = ExchangeAgent(
    id=0,
    mkt_open=0,
    mkt_close=1_000_000_000,  # Example value; adjust as needed
    symbols=["PEN"],
    name="TestExchange",
    type="ExchangeAgent",
    random_state=np.random.RandomState(seed=42),
    log_orders=False,
    use_metric_tracker=False
    )

    # Optionally, set up a fake order book if required
    exchange_agent.order_books["PEN"] = FakeOrderBook()

    random_state = np.random.RandomState(seed=42)


    value_agent = ValueAgent(
        id=agent_id,
        symbol=symbol,
        random_state=random_state,
        collateral=collateral,
        wake_up_freq=wake_up_freq,
        r_bar=r_bar,
        coef=coef
    )

    kernel = FakeKernel(
        agents=[exchange_agent, value_agent], # first one
        swap_interval=str_to_ns("8h"),
    )
    class MockRateOracle:
        def get_floating_rate(self, current_time):
            return 0.02  

    mock_rate_oracle = MockRateOracle()
    value_agent.rate_oracle = mock_rate_oracle
    value_agent.kernel = kernel
    exchange_agent.kernel = kernel

    value_agent.mkt_open = 1
    value_agent.mkt_close = 1_000_000_000  
    value_agent.current_time = 2

    value_agent.exchange_id = 0

    value_agent.known_bids = {symbol: [(1000, 50)]}
    value_agent.known_asks = {symbol: [(1010, 50)]}


    placed_orders = []
    logger.debug("Initialized list to capture placed orders")

    def mock_place_limit_order(symbol, quantity, side, price):
        logger.debug(f"Placed limit order - Symbol: {symbol}, Quantity: {quantity}, Side: {side}, Price: {price}")
        placed_orders.append({'symbol': symbol, 'quantity': quantity, 'side': side, 'price': price})

    value_agent.place_limit_order = mock_place_limit_order
    logger.debug("Replaced agent's place_limit_order method with mock method")


    def mock_observe_price(symbol, current_time, random_state):
        return 1005 

    def mock_get_floating_rate(current_time):
        return 0.02 

    kernel.driving_oracle = type('MockOracle', (), {'observe_price': mock_observe_price})
    kernel.rate_oracle = type('MockOracle', (), {'get_floating_rate': mock_get_floating_rate})

    logger.debug("Simulating agent's wakeup call")
    value_agent.wakeup(value_agent.current_time)

    logger.debug("Simulating receipt of QuerySpreadResponseMsg")
    message = QuerySpreadResponseMsg(
        symbol=symbol,
        bids=value_agent.known_bids[symbol],
        asks=value_agent.known_asks[symbol],
        mkt_closed=False,
        depth=1,
        last_trade=None,
    )
    
    class MockDrivingOracle:
        def observe_price(self, symbol, current_time, random_state):
            return 1005  


    mock_driving_oracle = MockDrivingOracle()


    value_agent.driving_oracle = mock_driving_oracle

    logger.debug("Agent receiving the QuerySpreadResponseMsg")
    value_agent.receive_message(value_agent.current_time, sender_id=0, message=message)

    assert len(placed_orders) == 1, f"Expected 1 order to be placed, but got {len(placed_orders)}"

    order = placed_orders[0]
    logger.debug(f"Order placed: {order}")


    assert order['side'] == Side.BID, f"Expected order side to be BID, but got {order['side']}"

    assert order['price'] == 1010, f"Expected order price to be 1010, but got {order['price']}"

    logger.info("ValueAgent test passed successfully.")

