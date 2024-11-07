import pytest
import numpy as np

from abides_core import Message, NanosecondTime
from abides_core.utils import str_to_ns, merge_swap
from abides_core.kernel import Kernel
from abides_markets.agents import TradingAgent, LiquidatorAgent
from abides_markets.orders import Side, MarketOrder
from abides_markets.messages.marketdata import MarketDataMsg, L2SubReqMsg
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.messages.order import MarketOrderMsg
from abides_markets.messages.market import MarketClosePriceRequestMsg

class FakeOrderBook:
    def __init__(self):
        self.twap = 1000
        self.bids = [(1000, 50), (990, 100)]
        self.asks = [(1010, 50), (1020, 100)]
    
    def get_twap(self):
        return self.twap
    
    def set_wakeup(self, agent_id: int, requested_time: NanosecondTime) -> None:
        pass

class FakeKernel:
    def __init__(self):
        self.book = FakeOrderBook()
        self.rate_normalizer = 1
        self.swap_interval = 1
        self.exchange_id = 0
    
    def send_message(self, sender_id, recipient_id, message, delay=0):
        pass

def test_liquidator_agent_initialization():
    agent_id = 1
    symbol = "PEN"
    wake_up_freq = str_to_ns("1h")
    collateral = 100000
    liquidator = LiquidatorAgent(
        id=agent_id,
        symbol=symbol,
        wake_up_freq=wake_up_freq,
        collateral=collateral,
    )
    
    assert liquidator.id == agent_id
    assert liquidator.symbol == symbol
    assert liquidator.wake_up_freq == wake_up_freq
    assert liquidator.position["COLLATERAL"] == collateral
    assert liquidator.state == "AWAITING_WAKEUP"
    assert liquidator.watch_list == []
    assert liquidator.failed_liquidation == 0

def test_liquidator_agent_wakeup():
    agent_id = 1
    symbol = "PEN"
    wake_up_freq = str_to_ns("1h")
    liquidator = LiquidatorAgent(
        id=agent_id,
        symbol=symbol,
        wake_up_freq=wake_up_freq,
    )
    current_time = 0

    liquidator.kernel = FakeKernel()
    liquidator.exchange_id = liquidator.kernel.exchange_id
    liquidator.mkt_open = True
    liquidator.mkt_closed = False
    liquidator.send_message = lambda recipient_id, message, delay=0: None
    liquidator.get_current_spread = lambda symbol: None

    liquidator.wakeup(current_time)
    assert liquidator.state == "AWAITING_SPREAD"

def test_liquidator_receive_message():
    agent_id = 1
    symbol = "PEN"
    liquidator = LiquidatorAgent(
        id=agent_id,
        symbol=symbol,
        subscribe=False,
    )
    current_time = 0
    liquidator.kernel = FakeKernel()
    liquidator.exchange_id = liquidator.kernel.exchange_id
    liquidator.state = "AWAITING_SPREAD"
    liquidator.mkt_open = True
    liquidator.mkt_closed = False

    watched_agent = TradingAgent(id=2)
    watched_agent.position = {"COLLATERAL": 1000, "SIZE": 100, "FIXRATE": 0.05}
    watched_agent.is_healthy = lambda: True
    liquidator.watch_list = [watched_agent]

    liquidator.known_bids = {symbol: [(990, 100)]}
    liquidator.known_asks = {symbol: [(1010, 100)]}

    message = QuerySpreadResponseMsg(
        symbol=symbol,
        bids=liquidator.known_bids[symbol],
        asks=liquidator.known_asks[symbol],
        mkt_closed=False,
        depth=1,
        last_trade=None,
    )

    liquidator.receive_message(current_time, sender_id=0, message=message)
    assert liquidator.failed_liquidation == 0
    assert liquidator.state == "AWAITING_WAKEUP"

def test_liquidator_check_liquidate_unhealthy_agent():
    liquidator = LiquidatorAgent(id=1)
    liquidator.kernel = FakeKernel()
    liquidator.exchange_id = liquidator.kernel.exchange_id
    liquidator.symbol = "PEN"
    liquidator.known_bids = {"PEN": [(1000, 50), (990, 100)]}
    liquidator.known_asks = {"PEN": [(1010, 50), (1020, 100)]}

    agent = TradingAgent(id=2)
    agent.kernel = liquidator.kernel
    agent.position = {"COLLATERAL": 10, "SIZE": 100, "FIXRATE": 0.05}
    agent.is_healthy = lambda: False
    agent.mRatio = lambda: 1.2
    agent.mark_to_market = lambda: 50
    agent.cancel_all_orders = lambda: None

    liquidator.place_market_order = lambda symbol, quantity, side: None

    result = liquidator.check_liquidate(agent)
    assert result
    assert agent.position["SIZE"] < 100
    assert liquidator.position["SIZE"] != 0
    assert liquidator.failed_liquidation >= 0

def test_liquidator_agent_insufficient_liquidity():
    liquidator = LiquidatorAgent(id=1)
    liquidator.kernel = FakeKernel()
    liquidator.exchange_id = liquidator.kernel.exchange_id
    liquidator.symbol = "PEN"
    liquidator.known_bids = {"PEN": [(1000, 10)]}
    liquidator.known_asks = {"PEN": [(1010, 10)]}

    agent = TradingAgent(id=2)
    agent.kernel = liquidator.kernel
    agent.position = {"COLLATERAL": 10, "SIZE": 100, "FIXRATE": 0.05}
    agent.is_healthy = lambda: False
    agent.mRatio = lambda: 1.2
    agent.mark_to_market = lambda: 50
    agent.cancel_all_orders = lambda: None

    liquidator.place_market_order = lambda symbol, quantity, side: None

    result = liquidator.check_liquidate(agent)
    assert result
    assert agent.position["SIZE"] == 90
    assert liquidator.failed_liquidation >= 0

def test_liquidator_agent_sell_option():
    liquidator = LiquidatorAgent(id=1)
    liquidator.kernel = FakeKernel()
    liquidator.exchange_id = liquidator.kernel.exchange_id
    liquidator.symbol = "PEN"
    liquidator.known_bids = {"PEN": [(1000, 50), (990, 100)]}
    liquidator.known_asks = {"PEN": [(1010, 50), (1020, 100)]}

    liquidator.place_market_order = lambda symbol, quantity, side: None

    agent = TradingAgent(id=2)
    agent.kernel = liquidator.kernel
    agent.position = {"COLLATERAL": 10, "SIZE": 100, "FIXRATE": 0.05}
    agent.is_healthy = lambda: False
    agent.mRatio = lambda: 1.2
    agent.mark_to_market = lambda: 50
    agent.cancel_all_orders = lambda: None

    result = liquidator.check_liquidate(agent, sell=True)
    assert result

def test_liquidator_agent_no_sell_option():
    liquidator = LiquidatorAgent(id=1)
    liquidator.kernel = FakeKernel()
    liquidator.exchange_id = liquidator.kernel.exchange_id
    liquidator.symbol = "PEN"
    liquidator.known_bids = {"PEN": [(1000, 50), (990, 100)]}
    liquidator.known_asks = {"PEN": [(1010, 50), (1020, 100)]}

    liquidator.place_market_order = lambda symbol, quantity, side: None

    agent = TradingAgent(id=2)
    agent.kernel = liquidator.kernel
    agent.position = {"COLLATERAL": 10, "SIZE": 100, "FIXRATE": 0.05}
    agent.is_healthy = lambda: False
    agent.mRatio = lambda: 1.2
    agent.mark_to_market = lambda: 50
    agent.cancel_all_orders = lambda: None

    result = liquidator.check_liquidate(agent, sell=False)
    assert result

def test_liquidator_agent_full_integration():
    liquidator = LiquidatorAgent(id=1)
    liquidator.kernel = FakeKernel()
    liquidator.exchange_id = liquidator.kernel.exchange_id
    liquidator.symbol = "PEN"
    liquidator.known_bids = {"PEN": [(1000, 50), (990, 100)]}
    liquidator.known_asks = {"PEN": [(1010, 50), (1020, 100)]}

    healthy_agent = TradingAgent(id=2)
    healthy_agent.kernel = liquidator.kernel
    healthy_agent.position = {"COLLATERAL": 1000, "SIZE": 100, "FIXRATE": 0.05}
    healthy_agent.is_healthy = lambda: True

    unhealthy_agent = TradingAgent(id=3)
    unhealthy_agent.kernel = liquidator.kernel
    unhealthy_agent.position = {"COLLATERAL": 10, "SIZE": 100, "FIXRATE": 0.05}
    unhealthy_agent.is_healthy = lambda: False
    unhealthy_agent.mRatio = lambda: 1.2
    unhealthy_agent.mark_to_market = lambda: 50
    unhealthy_agent.cancel_all_orders = lambda: None

    liquidator.place_market_order = lambda symbol, quantity, side: None
    liquidator.mkt_open = True
    liquidator.mkt_closed = False

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

    assert unhealthy_agent.position["SIZE"] < 100
    assert liquidator.position["SIZE"] != 0
