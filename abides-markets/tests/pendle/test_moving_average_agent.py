from turtle import title
import pytest
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd

from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from abides_markets.agents import MovingAverageAgent, ExchangeAgent
from abides_markets.messages.query import QueryLastTradeResponseMsg
from abides_core.kernel import Kernel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FakePriceProvider:
    def __init__(self, initial_price=100.0, seed=42):
        self.random_state = np.random.RandomState(seed)
        self.current_price = initial_price

    def get_next_price(self):
        self.current_price += self.random_state.normal(0, 1)
        return self.current_price

def plot_histogram(data, bins, title, xlabel, ylabel, filename, color='skyblue', alpha=0.7, edgecolor='black', show_stats=True):
    plt.figure(figsize=(10, 6))
    n, bin_edges, patches = plt.hist(data, bins=bins, alpha=alpha, color=color, edgecolor=edgecolor, density=False)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    if show_stats:
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data)
        plt.axvline(mean, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean:.2f}')
        plt.axvline(median, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median:.2f}')
        plt.axvline(mean + std, color='orange', linestyle='dashed', linewidth=1.5, label=f'+1 Std: {mean + std:.2f}')
        plt.axvline(mean - std, color='orange', linestyle='dashed', linewidth=1.5, label=f'-1 Std: {mean - std:.2f}')
        plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_price_ma_orders(title, prices, short_window, long_window, orders, filename):
    plt.figure(figsize=(10,6))
    plt.plot(prices, label='Price', alpha=0.7)
    if len(prices) >= long_window:
        ma_long = np.convolve(prices, np.ones(long_window)/long_window, mode='valid')
        ma_short = np.convolve(prices, np.ones(short_window)/short_window, mode='valid')
        offset = long_window - short_window
        if offset > 0:
            ma_short = ma_short[offset:]
        ma_long_series = np.full(len(prices), np.nan)
        ma_long_series[long_window-1:] = ma_long
        ma_short_series = np.full(len(prices), np.nan)
        ma_short_series[long_window-1:] = ma_short
        plt.plot(ma_long_series, label=f"MA {long_window}", color='orange', linewidth=2)
        plt.plot(ma_short_series, label=f"MA {short_window}", color='green', linewidth=2)
    if len(orders) > 0:
        def time_to_index(t):
            return int((t - 100_000_000) / 1_000_000)
        buy_orders = [o for o in orders if o['action'] == 'buy']
        sell_orders = [o for o in orders if o['action'] == 'sell']
        if len(buy_orders) > 0:
            buy_idx = [time_to_index(o['time']) for o in buy_orders if 0 <= time_to_index(o['time']) < len(prices)]
            plt.scatter(buy_idx, np.array(prices)[buy_idx], color='green', marker='^', label='Buy Orders')
        if len(sell_orders) > 0:
            sell_idx = [time_to_index(o['time']) for o in sell_orders if 0 <= time_to_index(o['time']) < len(prices)]
            plt.scatter(sell_idx, np.array(prices)[sell_idx], color='red', marker='v', label='Sell Orders')
    plt.title(f"{title}")
    plt.xlabel("Time Step (index)")
    plt.ylabel("Price ($)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def test_moving_average_agent_momentum():
    num_iterations = 500
    wakeup_times = []
    price_history = []
    output_dir = "./logs_ma/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    mkt_open = 100_000_000
    mkt_close = 600_000_000
    exchange_agent = ExchangeAgent(
        id=0,
        mkt_open=mkt_open,
        mkt_close=mkt_close,
        symbols=["PEN"],
        name="TestExchange",
        type="ExchangeAgent",
        random_state=np.random.RandomState(seed=42),
        log_orders=False,
        use_metric_tracker=False
    )
    ma_agent = MovingAverageAgent(
        id=1,
        symbol="PEN",
        random_state=np.random.RandomState(seed=43),
        short_window=20,
        long_window=50,
        strategy="momentum",
        test_mode=True,
        output_dir=output_dir
    )
    kernel = Kernel(
        agents=[exchange_agent, ma_agent],
        swap_interval=str_to_ns("8h")
    )
    ma_agent.kernel = kernel
    exchange_agent.kernel = kernel
    ma_agent.mkt_open = mkt_open
    ma_agent.mkt_close = mkt_close
    ma_agent.current_time = 0
    ma_agent.exchange_id = 0
    price_provider = FakePriceProvider(initial_price=100.0, seed=123)
    for i in range(num_iterations):
        current_time = mkt_open + i * 1_000_000
        if current_time >= mkt_close:
            current_time = mkt_close - 1
        ma_agent.current_time = current_time
        ma_agent.wakeup(current_time)
        last_trade_price = price_provider.get_next_price()
        price_history.append(last_trade_price)
        message = QueryLastTradeResponseMsg(
            symbol="PEN",
            mkt_closed=False,
            last_trade=last_trade_price
        )
        ma_agent.receive_message(current_time, sender_id=0, message=message)
        wakeup_times.append(current_time)
    ma_agent.kernel_stopping()
    intervals_file = os.path.join(output_dir, "wakeup_intervals.csv")
    orders_file = os.path.join(output_dir, "orders.csv")
    price_file = os.path.join(output_dir, "price_history.csv")
    assert os.path.exists(intervals_file)
    assert os.path.exists(orders_file)
    assert os.path.exists(price_file)
    intervals_df = pd.read_csv(intervals_file)
    orders_df = pd.read_csv(orders_file)
    price_df = pd.read_csv(price_file)
    assert len(price_df) == num_iterations
    assert len(intervals_df) == num_iterations - 1
    plot_histogram(
        data=intervals_df['wakeup_intervals'].values,
        bins=50,
        title='Distribution of MA Agent Wakeup Times(momentum)',
        xlabel='Wakeup Interval (ns)',
        ylabel='Frequency',
        filename=os.path.join(output_dir, "ma_agent_wakeup_intervals_momentum.png")
    )
    full_prices = price_df['price'].values
    orders_df['price'] = orders_df['price'] / 100.0
    orders_list = orders_df.to_dict('records')
    if len(full_prices) >= ma_agent.long_window:
        plot_price_ma_orders(
            title='Price, Moving Averages and Orders (momentum)',
            prices=full_prices,
            short_window=ma_agent.short_window,
            long_window=ma_agent.long_window,
            orders=orders_list,
            filename=os.path.join(output_dir, "ma_agent_price_ma_orders_momentum.png")
        )
    assert len(wakeup_times) == num_iterations
    logger.info("MovingAverageAgent test completed successfully. Check ./logs_ma/ for results.")

def test_moving_average_agent_mean_reversion():
    num_iterations = 500
    wakeup_times = []
    price_history = []
    output_dir = "./logs_ma/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    mkt_open = 100_000_000
    mkt_close = 600_000_000
    exchange_agent = ExchangeAgent(
        id=0,
        mkt_open=mkt_open,
        mkt_close=mkt_close,
        symbols=["PEN"],
        name="TestExchange",
        type="ExchangeAgent",
        random_state=np.random.RandomState(seed=42),
        log_orders=False,
        use_metric_tracker=False
    )
    ma_agent = MovingAverageAgent(
        id=1,
        symbol="PEN",
        random_state=np.random.RandomState(seed=43),
        short_window=20,
        long_window=50,
        strategy="mean_reversion",
        test_mode=True,
        output_dir=output_dir
    )
    kernel = Kernel(
        agents=[exchange_agent, ma_agent],
        swap_interval=str_to_ns("8h")
    )
    ma_agent.kernel = kernel
    exchange_agent.kernel = kernel
    ma_agent.mkt_open = mkt_open
    ma_agent.mkt_close = mkt_close
    ma_agent.current_time = 0
    ma_agent.exchange_id = 0
    price_provider = FakePriceProvider(initial_price=100.0, seed=123)
    for i in range(num_iterations):
        current_time = mkt_open + i * 1_000_000
        if current_time >= mkt_close:
            current_time = mkt_close - 1
        ma_agent.current_time = current_time
        ma_agent.wakeup(current_time)
        last_trade_price = price_provider.get_next_price()
        price_history.append(last_trade_price)
        message = QueryLastTradeResponseMsg(
            symbol="PEN",
            mkt_closed=False,
            last_trade=last_trade_price
        )
        ma_agent.receive_message(current_time, sender_id=0, message=message)
        wakeup_times.append(current_time)
    ma_agent.kernel_stopping()
    intervals_file = os.path.join(output_dir, "wakeup_intervals.csv")
    orders_file = os.path.join(output_dir, "orders.csv")
    price_file = os.path.join(output_dir, "price_history.csv")
    assert os.path.exists(intervals_file)
    assert os.path.exists(orders_file)
    assert os.path.exists(price_file)
    intervals_df = pd.read_csv(intervals_file)
    orders_df = pd.read_csv(orders_file)
    price_df = pd.read_csv(price_file)
    assert len(price_df) == num_iterations
    assert len(intervals_df) == num_iterations - 1
    plot_histogram(
        data=intervals_df['wakeup_intervals'].values,
        bins=50,
        title='Distribution of MA Agent Wakeup Times(mean_reversion)',
        xlabel='Wakeup Interval (ns)',
        ylabel='Frequency',
        filename=os.path.join(output_dir, "ma_agent_wakeup_intervals_mean_reversion.png")
    )
    full_prices = price_df['price'].values
    orders_df['price'] = orders_df['price'] / 100.0
    orders_list = orders_df.to_dict('records')
    if len(full_prices) >= ma_agent.long_window:
        plot_price_ma_orders(
            title='Price, Moving Averages and Orders (mean_reversion)',
            prices=full_prices,
            short_window=ma_agent.short_window,
            long_window=ma_agent.long_window,
            orders=orders_list,
            filename=os.path.join(output_dir, "ma_agent_price_ma_orders_mean_reversion.png")
        )
    assert len(wakeup_times) == num_iterations
    logger.info("MovingAverageAgent test completed successfully. Check ./logs_ma/ for results.")