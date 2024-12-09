import pytest
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

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

class OrderSizeModel:
    def sample(self, random_state):
        return random_state.randint(1, 100)

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
    """
    Test the NoiseAgent algorithm to validate wakeup times and order size distributions.
    """
    num_iterations = 1000  
    wakeup_times = []
    order_sizes = []

    os.makedirs('./logs/', exist_ok=True)

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
    exchange_agent.order_books["PEN"] = FakeOrderBook()

    noise_agent = NoiseAgent(
        id=1,
        symbol="PEN",
        random_state=np.random.RandomState(seed=42),
        collateral=100_000,
        wakeup_time=500_000_000,
    )

    kernel = Kernel(
        agents=[exchange_agent, noise_agent],
        swap_interval=str_to_ns("8h"),
    )
    noise_agent.kernel = kernel
    exchange_agent.kernel = kernel

    noise_agent.mkt_open = 1
    noise_agent.mkt_close = 1_000_000_000
    noise_agent.current_time = 0
    noise_agent.exchange_id = 0

    noise_agent.known_bids = {"PEN": [(1000, 50)]}
    noise_agent.known_asks = {"PEN": [(1010, 50)]}

    noise_agent.order_size_model = OrderSizeModel()

    for i in range(num_iterations):
        current_time = 500_000_000 + i * 1_000_000  
        noise_agent.current_time = current_time
        noise_agent.wakeup(current_time)

        message = QuerySpreadResponseMsg(
            symbol="PEN",
            bids=noise_agent.known_bids["PEN"],
            asks=noise_agent.known_asks["PEN"],
            mkt_closed=False,
            depth=1,
            last_trade=None,
        )
        noise_agent.receive_message(current_time, sender_id=0, message=message)

        wakeup_times.append(current_time)

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

    assert 0 < mean_order_size < 1000, "Mean order size out of expected range."
    assert 0 < std_order_size < 500, "Std dev of order size out of expected range."

    logger.info("NoiseAgent algorithm test passed.")