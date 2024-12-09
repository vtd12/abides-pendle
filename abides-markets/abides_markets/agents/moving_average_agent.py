import logging
from typing import Optional, List
import numpy as np
import os
import pandas as pd

from abides_core import Message, NanosecondTime
from abides_markets.messages.query import QueryLastTradeResponseMsg
from abides_markets.orders import Side
from .trading_agent import TradingAgent
from ..generators import OrderSizeGenerator

logger = logging.getLogger(__name__)

class MovingAverageAgent(TradingAgent):
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
        strategy: str = "momentum",
        test_mode: bool = False,
        output_dir: str = "./output"
    ) -> None:
        super().__init__(id, name, type, random_state, symbol, log_orders, collateral)
        self.wakeup_time: NanosecondTime = wakeup_time
        self.symbol: str = symbol
        self.state: str = "AWAITING_WAKEUP"
        self.size: Optional[int] = (
            self.random_state.randint(20, 50) if order_size_model is None else None
        )
        self.order_size_model = order_size_model
        self.short_window = short_window
        self.long_window = long_window
        self.strategy = strategy.lower()
        assert self.strategy in ["momentum", "mean_reversion"]
        self.trade_history: List[float] = []
        self.test_mode = test_mode
        self.output_dir = output_dir
        self.prev_wakeup_time: Optional[NanosecondTime] = None
        self.wakeup_intervals: List[int] = []
        self.orders_placed: List[dict] = []
        if self.test_mode and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def kernel_starting(self, start_time: NanosecondTime) -> None:
        super().kernel_starting(start_time)
        logger.debug(f"{self.name} initialized with strategy: {self.strategy}")

    def wakeup(self, current_time: NanosecondTime) -> None:
        if self.test_mode and self.prev_wakeup_time is not None:
            interval = current_time - self.prev_wakeup_time
            self.wakeup_intervals.append(interval)
            logger.debug(f"{self.name} wakeup interval: {interval} ns")
        self.prev_wakeup_time = current_time
        super().wakeup(current_time)
        self.state = "AWAITING_WAKEUP"
        if not (self.mkt_open <= current_time < self.mkt_close):
            logger.debug(f"{self.name} detected market is closed at time {current_time}.")
            return
        if not hasattr(self, 'trading') or not self.trading:
            self.trading = True
            logger.debug(f"{self.name} is ready to start trading.")
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            logger.debug(f"{self.name} detected market closed, stopping trading.")
            return
        if self.wakeup_time and self.wakeup_time > current_time:
            self.set_wakeup(self.wakeup_time)
            logger.debug(f"{self.name} set next wakeup at {self.wakeup_time}.")
            return
        self.get_last_trade(self.symbol)
        self.state = "AWAITING_LAST_TRADE"
        logger.debug(f"{self.name} requested last trade for {self.symbol} at time {current_time}.")

    def place_order(self, action: str, price: float) -> None:
        if self.order_size_model is not None:
            self.size = self.order_size_model.sample(random_state=self.random_state)
        if self.size and self.size > 0:
            order_side = Side.BID if action == "buy" else Side.ASK
            price_cents = int(round(price * 100))
            assert isinstance(price_cents, int), f"price_cents should be int, got {type(price_cents)}"
            self.place_limit_order(self.symbol, self.size, order_side, price_cents)
            logger.info(f"{self.name} placed {action.upper()} order at ${price:.2f} for size {self.size}")
            if self.test_mode:
                self.orders_placed.append({
                    "time": self.current_time,
                    "action": action,
                    "price": price_cents,
                    "size": self.size
                })
                logger.debug(f"Order recorded: {self.orders_placed[-1]}")

    def receive_message(self, current_time: NanosecondTime, sender_id: int, message: Message) -> None:
        super().receive_message(current_time, sender_id, message)
        if self.state == "AWAITING_LAST_TRADE":
            if isinstance(message, QueryLastTradeResponseMsg):
                last_trade_price = self.last_trade[self.symbol]
                logger.debug(f"{self.name} received last trade price: {last_trade_price}")
                self.trade_history.append(last_trade_price)
                logger.debug(f"Trade history updated: {len(self.trade_history)} entries")
                if len(self.trade_history) > self.long_window and not self.test_mode:
                    removed_price = self.trade_history.pop(0)
                    logger.debug(f"{self.name} removed oldest trade price: {removed_price}")
                if len(self.trade_history) >= self.long_window:
                    self.evaluate_strategy(current_time)
                else:
                    logger.debug(f"{self.name} waiting for more trade data.")
                self.state = "AWAITING_WAKEUP"

    def evaluate_strategy(self, current_time: NanosecondTime) -> None:
        if len(self.trade_history) < self.long_window:
            logger.debug(f"{self.name} waiting for more trade data.")
            return
        prices = self.trade_history[-self.long_window:]
        ma_short = np.mean(prices[-self.short_window:])
        ma_long = np.mean(prices)
        last_price = prices[-1]
        logger.debug(f"{self.name} - MA{self.short_window}: {ma_short}, MA{self.long_window}: {ma_long}")
        action = None
        if self.strategy == "momentum":
            if ma_short > ma_long:
                action = "buy"
            elif ma_short < ma_long:
                action = "sell"
        elif self.strategy == "mean_reversion":
            if ma_short > ma_long:
                action = "sell"
            elif ma_short < ma_long:
                action = "buy"
        if action is not None:
            self.place_order(action, last_price)
        next_wakeup = current_time + self.get_wake_frequency()
        self.set_wakeup(next_wakeup)
        logger.debug(f"{self.name} set next wakeup at {next_wakeup}.")

    def get_wake_frequency(self) -> NanosecondTime:
        return self.random_state.randint(low=1_000_000, high=10_000_000)

    def kernel_stopping(self) -> None:
        if self.test_mode:
            intervals_df = pd.DataFrame({"wakeup_intervals": self.wakeup_intervals})
            intervals_df.to_csv(os.path.join(self.output_dir, "wakeup_intervals.csv"), index=False)
            logger.debug(f"wakeup_intervals.csv written with {len(intervals_df)} entries.")
            orders_df = pd.DataFrame(self.orders_placed)
            orders_df.to_csv(os.path.join(self.output_dir, "orders.csv"), index=False)
            logger.debug(f"orders.csv written with {len(orders_df)} entries.")
            price_history_df = pd.DataFrame({"price": self.trade_history})
            price_history_df.to_csv(os.path.join(self.output_dir, "price_history.csv"), index=False)
            logger.debug(f"price_history.csv written with {len(price_history_df)} entries.")
        super().kernel_stopping()
