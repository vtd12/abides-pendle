import logging
from typing import Optional, List

import numpy as np
from abides_core import Message, NanosecondTime
from ..messages import QueryTradeHistoryResponseMsg, QueryLastTradeResponseMsg
from ..orders import Side

from .trading_agent import TradingAgent
from ..generators import OrderSizeGenerator

logger = logging.getLogger(__name__)

class MovingAverageAgent(TradingAgent):
    """
    Moving Average Agent implements a strategy based on two MA curve.
    It can act as a Momentum Agent (following the trend) or a Mean-Reversion Agent (reversing the trend)
    based on the comparison of short-term and long-term moving averages.
    (e.g. MA20 and MA50)
    """

    def __init__(
        self,
        id: int,
        name: Optional[str] = None,
        type: Optional[str] = None,
        random_state: Optional[np.random.RandomState] = None,
        symbol: str = "PEN",
        log_orders: bool = False,
        collateral: int = 100000,
        order_size_model: Optional[OrderSizeGenerator] = None,
        wakeup_time: Optional[NanosecondTime] = None,
        short_window: int = 20,
        long_window: int = 50,
        strategy: str = "momentum",  # or "mean_reversion"
    ) -> None:
        super().__init__(id, name, type, random_state, symbol, log_orders, collateral)

        self.wakeup_time: NanosecondTime = wakeup_time
        self.symbol: str = symbol
        self.trading: bool = False
        self.state: str = "AWAITING_WAKEUP"
        self.size: Optional[int] = (
            self.random_state.randint(20, 50) if order_size_model is None else None
        )
        self.order_size_model = order_size_model

        # MA agent parameters
        self.short_window = short_window
        self.long_window = long_window
        self.strategy = strategy.lower()
        assert self.strategy in ["momentum", "mean_reversion"], "Strategy must be 'momentum' or 'mean_reversion'"

        # Trade history storage
        self.trade_history: List[float] = []

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)
        logger.debug(f"{self.name} initialized with strategy: {self.strategy}")

    def wakeup(self, current_time: NanosecondTime) -> None:
        super().wakeup(current_time)
        self.state = "INACTIVE"

        if not self.mkt_open or not self.mkt_close:
            return
        else:
            if not self.trading:
                self.trading = True
                logger.debug(f"{self.name} is ready to start trading now.")

        if self.mkt_closed and (self.symbol in self.daily_close_price):
            return

        if self.wakeup_time and self.wakeup_time > current_time:
            self.set_wakeup(self.wakeup_time)
            return

        if self.mkt_closed and (self.symbol not in self.daily_close_price):
            self.get_current_spread(self.symbol)
            self.state = "AWAITING_SPREAD"
            return

        # Request trade history to compute moving averages
        self.get_trade_history(self.symbol, self.long_window)
        self.state = "AWAITING_TRADE_HISTORY"

    def place_order(self, action: str, price: float) -> None:
        """
        Place a limit order based on the action and price.
        """
        if self.order_size_model is not None:
            self.size = self.order_size_model.sample(random_state=self.random_state)

        if self.size and self.size > 0:
            if action == "buy":
                self.place_limit_order(self.symbol, self.size, Side.BID, price)
                logger.info(f"{self.name} placed BUY order at {price} for size {self.size}")
            elif action == "sell":
                self.place_limit_order(self.symbol, self.size, Side.ASK, price)
                logger.info(f"{self.name} placed SELL order at {price} for size {self.size}")

    def receive_message(
        self, current_time: NanosecondTime, sender_id: int, message: Message
    ) -> None:
        """
        Handle incoming messages based on the current state.
        """
        super().receive_message(current_time, sender_id, message)

        if self.state == "AWAITING_TRADE_HISTORY":
            if isinstance(message, QueryTradeHistoryResponseMsg):
                trades = message.trade_history
                self.trade_history.extend([trade.price for trade in trades])
                self.trade_history = self.trade_history[-self.long_window:]
                if len(self.trade_history) >= self.long_window:
                    self.evaluate_strategy(current_time)
                self.state = "AWAITING_WAKEUP"

        elif self.state == "AWAITING_SPREAD":
            if isinstance(message, QueryLastTradeResponseMsg):
                last_trade_price = self.last_trade[self.symbol]
                self.evaluate_spread_strategy(current_time, last_trade_price)
                self.state = "AWAITING_WAKEUP"

    def evaluate_strategy(self, current_time: NanosecondTime) -> None:
        """
        Evaluate the trading strategy based on moving averages.
        """
        if len(self.trade_history) < self.long_window:
            logger.debug(f"{self.name} waiting for more trade data.")
            return

        prices = self.trade_history[-self.long_window:]
        ma_short = np.mean(prices[-self.short_window:])
        ma_long = np.mean(prices)

        logger.debug(
            f"{self.name} - MA{self.short_window}: {ma_short}, MA{self.long_window}: {ma_long}"
        )

        last_price = prices[-1]

        if self.strategy == "momentum":
            if ma_short > ma_long:
                # Uptrend: place a buy order
                self.place_order("buy", last_price)
            elif ma_short < ma_long:
                # Downtrend: place a sell order
                self.place_order("sell", last_price)

        elif self.strategy == "mean_reversion":
            if ma_short > ma_long:
                # Overbought: place a sell order
                self.place_order("sell", last_price)
            elif ma_short < ma_long:
                # Oversold: place a buy order
                self.place_order("buy", last_price)

        # Schedule next wakeup
        next_wakeup = current_time + self.get_wake_frequency()
        self.set_wakeup(next_wakeup)

    def evaluate_spread_strategy(self, current_time: NanosecondTime, last_trade_price: float) -> None:
        """
        Evaluate and place orders based on the current spread.
        """
        buy_indicator = self.random_state.randint(0, 2)  # 0 or 1

        bid, bid_vol, ask, ask_vol = self.get_known_bid_ask(self.symbol)

        if self.order_size_model is not None:
            self.size = self.order_size_model.sample(random_state=self.random_state)

        if self.size and self.size > 0:
            if buy_indicator == 1 and ask:
                self.place_limit_order(self.symbol, self.size, Side.BID, ask)
                logger.info(f"{self.name} placed BUY order at {ask} for size {self.size}")
            elif buy_indicator == 0 and bid:
                self.place_limit_order(self.symbol, self.size, Side.ASK, bid)
                logger.info(f"{self.name} placed SELL order at {bid} for size {self.size}")

    def get_wake_frequency(self) -> NanosecondTime:
        """
        Define the frequency of wakeups in nanoseconds.
        """
        return self.random_state.randint(low=1_000_000, high=10_000_000)