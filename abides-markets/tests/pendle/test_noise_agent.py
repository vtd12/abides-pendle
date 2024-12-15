from re import L
from py import log
import pytest
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from abides_markets.agents import NoiseAgent, ExchangeAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.messages.market import MarketHoursMsg
from abides_markets.orders import Side
from abides_markets.agents.utils import tick_to_rate
from abides_core.kernel import Kernel
from abides_markets.order_book import OrderBook
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)


class FakeRateOracle:
    def __init__(self):
        pass

    def get_floating_rate(self, current_time: NanosecondTime) -> float:
        return tick_to_rate(1005)

class OrderSizeModel:
    def sample(self, random_state):
        return random_state.rand() 

def plot_histogram(data, bins, title, xlabel, ylabel, filename, color='skyblue', alpha=0.7, edgecolor='black', show_stats=True):
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(data, bins=bins, alpha=alpha, color=color, edgecolor=edgecolor, density=False)
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

def test_noise_agent_algorithm():
    num_iterations = 2 
    wakeup_times = []
    order_sizes = []

    os.makedirs('./logs/', exist_ok=True)

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
    exchange_agent.order_books["PEN"] = OrderBook(owner=exchange_agent, symbol="PEN")

    noise_agent = NoiseAgent(
        id=1,
        symbol="PEN",
        random_state=np.random.RandomState(seed=42),  
        collateral=100000,  
        wakeup_time=str_to_ns("1d"),
        log_orders=True 
    )

    kernel = Kernel(
        agents=[exchange_agent, noise_agent],
        swap_interval=str_to_ns("8h"),
    )
    noise_agent.kernel = kernel
    exchange_agent.kernel = kernel

    noise_agent.mkt_open = 1
    noise_agent.mkt_close = 365 * str_to_ns("1d")
    noise_agent.current_time = 2
    noise_agent.exchange_id = 0

    noise_agent.known_bids = {"PEN": [(1000, 1090001)]} 
    noise_agent.known_asks = {"PEN": [(1010, 1090142)]}
    noise_agent.order_size_model = OrderSizeModel()

    for i in range(num_iterations):
        current_time = str_to_ns("1d") + i * str_to_ns("1h")  
        noise_agent.current_time = current_time
        noise_agent.kernel_starting(current_time)
        noise_agent.wakeup(current_time)
        logger.debug(f"wake_up_success = {noise_agent.wake_up_success}")
        logger.debug(f"noise_agent.state = {noise_agent.state}")
        market_hours_msg = MarketHoursMsg(
            mkt_open=noise_agent.mkt_open,
            mkt_close=noise_agent.mkt_close
        )
        noise_agent.receive_message(
            current_time=current_time,
            sender_id=exchange_agent.id,
            message=market_hours_msg
        )
        # agent state 
        logger.debug(f"noise_agent.state = {noise_agent.state}")
        message = QuerySpreadResponseMsg(
            symbol="PEN",
            bids=noise_agent.known_bids["PEN"],
            asks=noise_agent.known_asks["PEN"],
            mkt_closed=False,
            depth=1,
            last_trade=1005,
        )
        noise_agent.receive_message(current_time, sender_id=0, message=message)
        
        logger.debug(f"noise_agent.size = {noise_agent.size}")
        
        wakeup_times.append(current_time)

        logger.debug(f"orders = {noise_agent.orders}")
        logger.debug(f"len(noise_agent.orders) = {len(noise_agent.orders)}")
        
        if len(noise_agent.orders) > 0:
            order = next(iter(noise_agent.orders.values()))
            order_sizes.append(order.quantity)
            noise_agent.orders.clear()

    histogram_params = {
        'bins': 50,
        'color': 'skyblue',
        'alpha': 0.7,
        'edgecolor': 'black',
        'show_stats': True
    }

    logger.debug("Starting to plot wakeup time distribution.")
    plot_histogram(
        data=wakeup_times,
        bins=histogram_params['bins'],
        title='Distribution of NoiseAgent Wakeup Times',
        xlabel='Time (Nanoseconds)',
        ylabel='Frequency',
        filename='./logs/noise_agent_wakeup_times.png',
        color=histogram_params['color'],
        alpha=histogram_params['alpha'],
        edgecolor=histogram_params['edgecolor'],
        show_stats=histogram_params['show_stats']
    )
    logger.debug("Wakeup time distribution plot saved as ./logs/noise_agent_wakeup_times.png")

    logger.debug("Starting to plot order size distribution.")
    plot_histogram(
        data=order_sizes,
        bins=histogram_params['bins'],
        title='Distribution of NoiseAgent Order Sizes',
        xlabel='Order Size',
        ylabel='Frequency',
        filename='./logs/noise_agent_order_sizes.png',
        color=histogram_params['color'],
        alpha=histogram_params['alpha'],
        edgecolor=histogram_params['edgecolor'],
        show_stats=histogram_params['show_stats']
    )
    logger.debug("Order size distribution plot saved as ./logs/noise_agent_order_sizes.png")

    assert len(wakeup_times) == num_iterations, f"Expected {num_iterations} wakeup times but got {len(wakeup_times)}"
    assert len(order_sizes) > 0, "No orders generated."

    mean_order_size = np.mean(order_sizes)
    std_order_size = np.std(order_sizes)
    logger.debug(f"Mean Order Size: {mean_order_size}")
    logger.debug(f"Std Dev of Order Size: {std_order_size}")

    logger.info("NoiseAgent algorithm test passed.")
