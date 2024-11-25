import pytest
import logging
import numpy as np
from unittest.mock import patch
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from abides_markets.agents import NoiseAgent, ExchangeAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from abides_core.kernel import Kernel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FakeOrderBook:
    def __init__(self):
        self.last_twap = -3.100998

    def get_twap(self):
        return self.last_twap

    def set_wakeup(self, agent_id: int, requested_time: NanosecondTime) -> None:
        pass

class FakeRateOracle:
    def __init__(self):
        pass

    def get_floating_rate(self, current_time: NanosecondTime) -> float:
        return -0.0003231358432291831

def test_noise_agent():
    """
    Test the NoiseAgent's behavior, ensuring it correctly places orders based on market data.
    """
    logger.debug("Starting test_noise_agent")

    logger.debug("Initializing ExchangeAgent")
    exchange_agent = ExchangeAgent(
        id=0,
        mkt_open=0,
        mkt_close=1_000_000_000,  
        symbols=["PEN"],
        name="TestExchange",
        type="ExchangeAgent",
        random_state=np.random.RandomState(seed=42),
        log_orders=False,
        use_metric_tracker=False
    )

    logger.debug("Setting FakeOrderBook for ExchangeAgent")
    exchange_agent.order_books["PEN"] = FakeOrderBook()


    agent_id = 1
    symbol = "PEN"
    collateral = 100_000
    wakeup_time = 500_000_000  

    logger.debug("Initializing NoiseAgent")
    random_state = np.random.RandomState(seed=42)
    noise_agent = NoiseAgent(
        id=agent_id,
        symbol=symbol,
        random_state=random_state,
        collateral=collateral,
        wakeup_time=wakeup_time,
    )

    logger.debug("Creating and configuring Kernel with ExchangeAgent and NoiseAgent")
    kernel = Kernel(
        agents=[exchange_agent, noise_agent],
        swap_interval=str_to_ns("8h"), 
    )

    logger.debug("Assigning kernel to agents")
    noise_agent.kernel = kernel
    exchange_agent.kernel = kernel

    logger.debug("Setting up NoiseAgent attributes")
    noise_agent.mkt_open = 1
    noise_agent.mkt_close = 1_000_000_000
    noise_agent.current_time = 0
    noise_agent.exchange_id = 0  

    logger.debug("Setting known bids and asks for NoiseAgent")
    noise_agent.known_bids = {symbol: [(1000, 50)]}
    noise_agent.known_asks = {symbol: [(1010, 50)]}

    placed_orders = []
    logger.debug("Initialized list to capture placed orders")

    def mock_place_limit_order(symbol, quantity, side, price):
        logger.debug(f"Placed limit order - Symbol: {symbol}, Quantity: {quantity}, Side: {side}, Price: {price}")
        placed_orders.append({'symbol': symbol, 'quantity': quantity, 'side': side, 'price': price})

    noise_agent.place_limit_order = mock_place_limit_order
    logger.debug("Replaced NoiseAgent's place_limit_order method with mock method")


    logger.debug("Simulating NoiseAgent's wakeup call")
    noise_agent.current_time = wakeup_time
    noise_agent.wakeup(noise_agent.current_time)


    logger.debug("Creating QuerySpreadResponseMsg")
    message = QuerySpreadResponseMsg(
        symbol=symbol,
        bids=noise_agent.known_bids[symbol],
        asks=noise_agent.known_asks[symbol],
        mkt_closed=False,
        depth=1,
        last_trade=None,
    )

    with patch('abides_markets.agents.value_agent.tick_to_rate') as mock_tick_to_rate:
        mock_tick_to_rate.return_value = 0.065 # >= 0.064

        logger.debug("Agent receiving the QuerySpreadResponseMsg")
        noise_agent.receive_message(noise_agent.current_time, sender_id=0, message=message)

    logger.debug("Asserting the number of placed orders")
    assert len(placed_orders) == 1, f"Expected 1 order to be placed, but got {len(placed_orders)}"

    order = placed_orders[0]
    logger.debug(f"Order placed: {order}")

    assert order['symbol'] == symbol, f"Expected order symbol to be {symbol}, but got {order['symbol']}"
    assert order['quantity'] == noise_agent.size, f"Expected order quantity to be {noise_agent.size}, but got {order['quantity']}"
    assert order['side'] in [Side.BID, Side.ASK], f"Expected order side to be BID or ASK, but got {order['side']}"
    assert order['price'] in [1000, 1010], f"Expected order price to be 1000 or 1010, but got {order['price']}"

    logger.info("NoiseAgent test passed successfully.")
