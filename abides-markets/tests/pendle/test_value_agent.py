import logging
import os
import numpy as np
from abides_core.utils import str_to_ns
from abides_core.kernel import Kernel
from abides_markets.agents import ValueAgent, ExchangeAgent
from abides_markets.order_book import OrderBook
from abides_markets.messages.query import QuerySpreadResponseMsg, QuerySpreadMsg
from abides_markets.messages.market import MarketHoursMsg
from abides_markets.orders import Side
from abides_markets.agents.utils import tick_to_rate, rate_to_tick
from abides_markets.models import OrderSizeModel

import types
from scipy.stats import expon, kstest

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

from abides_core import Message, NanosecondTime

class FakeRateOracle:
    def __init__(self, floating_rate):
        self.floating_rate = floating_rate

    def get_floating_rate(self, current_time) -> float:
        return self.floating_rate

class FakeDrivingOracle:
    def __init__(self, observed_rates):
        self.observed_rates = observed_rates
        self.index = 0

    def observe_price(self, symbol, current_time, random_state=None):
        if self.index < len(self.observed_rates):
            rate = self.observed_rates[self.index]
            self.index += 1
            return rate
        return self.observed_rates[-1]

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

def setup_agents_and_kernel(observed_rates, floating_rate, expected_sizes, mkt_close=None, exchange_agent_class=ExchangeAgent):
    exchange_agent = exchange_agent_class(
        id=0,
        mkt_open=0,
        mkt_close=mkt_close if mkt_close else 365 * str_to_ns("1d"),
        symbols=["PEN"],
        name="TestExchange",
        type="ExchangeAgent",
        random_state=np.random.RandomState(seed=42),
        log_orders=False,
        use_metric_tracker=True,
    )
    # 初始化实际的 OrderBook
    exchange_agent.order_books["PEN"] = OrderBook(owner=exchange_agent, symbol="PEN")

    driving_oracle = FakeDrivingOracle(observed_rates=observed_rates)
    rate_oracle = FakeRateOracle(floating_rate=floating_rate)

    order_size_model = OrderSizeModel()

    value_agent = ValueAgent(
        id=1,
        symbol="PEN",
        random_state=np.random.RandomState(seed=42),
        collateral=100_000,
        wake_up_freq=str_to_ns("10sec"),  # 调整为 10 秒以便测试
        r_bar=0.10,
        coef=[0.05, 0.40],
        order_size_model=order_size_model
    )

    kernel = Kernel(
        agents=[exchange_agent, value_agent],
        swap_interval=str_to_ns("1sec"),  # 设置为1秒，以确保事件被正确处理
    )
    kernel.driving_oracle = driving_oracle
    kernel.rate_oracle = rate_oracle
    value_agent.kernel = kernel
    exchange_agent.kernel = kernel

    value_agent.mkt_open = 1
    value_agent.mkt_close = mkt_close if mkt_close else 365 * str_to_ns('1d')
    value_agent.current_time = 0 
    value_agent.exchange_id = 0

    known_bids = {"PEN": [(1000, 50)]}    
    known_asks = {"PEN": [(1100, 50)]}    
    value_agent.known_bids = known_bids
    value_agent.known_asks = known_asks

    return exchange_agent, value_agent, kernel, known_bids, known_asks

def test_value_agent_calculation_logic():
    num_iterations = 10 
    expected_sizes = [271188] * num_iterations 

    os.makedirs('./logs/', exist_ok=True)

    observed_rates = [1028] * num_iterations  
    floating_rate = 0.1095
    mkt_close_time = str_to_ns("100sec") 

    exchange_agent, value_agent, kernel, known_bids, known_asks = setup_agents_and_kernel(
        observed_rates=observed_rates,
        floating_rate=floating_rate,
        expected_sizes=expected_sizes,
        mkt_close=mkt_close_time,
        exchange_agent_class=ExchangeAgent  
    )

    order_sizes = []
    decision_logs = []
    r_t_history = [] 
    mid_rate_history = [] 

    for i in range(num_iterations):
        current_time = 0 + i * str_to_ns("10sec")  
        value_agent.current_time = current_time
        value_agent.kernel_starting(current_time)
        value_agent.wakeup(current_time)
        
        market_hours_msg = MarketHoursMsg(
            mkt_open=value_agent.mkt_open,
            mkt_close=value_agent.mkt_close
        )
        value_agent.receive_message(
            current_time=current_time,
            sender_id=exchange_agent.id,
            message=market_hours_msg
        )

        value_agent.wakeup(current_time)  

        
        if len(value_agent.orders) > 0:
            order = next(iter(value_agent.orders.values()))
            order_sizes.append(order.quantity)
            decision_logs.append({
                'decision': order.side,
                'price': getattr(order, 'limit_price', None),  
                'quantity': order.quantity,
            })
            r_t_history.append(value_agent.r_t)
            bid, bid_vol = known_bids["PEN"][0]
            ask, ask_vol = known_asks["PEN"][0]
            mid_tick = (ask + bid) / 2  
            mid_rate = tick_to_rate(mid_tick)  
            mid_rate_history.append(mid_rate)
            value_agent.orders.clear()
        else:
            r_t_history.append(value_agent.r_t)
            bid, bid_vol = known_bids["PEN"][0]
            ask, ask_vol = known_asks["PEN"][0]
            mid_tick = (ask + bid) / 2 
            mid_rate = tick_to_rate(mid_tick)  
            mid_rate_history.append(mid_rate)

    expected_decisions = []
    for i in range(num_iterations):
        current_r_t = r_t_history[i]
        mid_rate = mid_rate_history[i]
        buy = current_r_t >= mid_rate
        expected_decisions.append(buy)

    for i in range(num_iterations):
        logger.debug(f"Iteration {i}: r_t={r_t_history[i]}, mid_rate={mid_rate_history[i]}, expected_buy={expected_decisions[i]}")

    for i in range(len(order_sizes)):
        log = decision_logs[i]
        expected_buy = expected_decisions[i]
        actual_buy = log['decision'] == Side.BID  

        assert actual_buy == expected_buy, f"Iteration {i}: Expected buy={expected_buy}, but got buy={actual_buy}"
        
        expected_p = 1100 if expected_buy else 1000  
        actual_p = log['price']
        assert actual_p == expected_p, f"Iteration {i}: Expected price={expected_p}, but got price={actual_p}"

    for i, size in enumerate(order_sizes):
        expected_size = expected_sizes[i]
        assert size == expected_size, f"Iteration {i}: Expected size={expected_size}, but got size={size}"
        
    histogram_params = {
        'bins': 10,
        'color': 'skyblue',
        'alpha': 0.7,
        'edgecolor': 'black',
        'show_stats': True
    }

    logger.debug("Starting to plot ValueAgent calculation logic debug information.")
    plot_histogram(
        data=r_t_history,
        bins=histogram_params['bins'],
        title='ValueAgent r_t Distribution',
        xlabel='r_t',
        ylabel='Frequency',
        filename='./logs/value_agent_r_t_distribution.png',
        color=histogram_params['color'],
        alpha=histogram_params['alpha'],
        edgecolor=histogram_params['edgecolor'],
        show_stats=histogram_params['show_stats']
    )
    logger.debug("r_t distribution saved as ./logs/value_agent_r_t_distribution.png")

    logger.info("ValueAgent calculation logic test passed successfully.")

def test_value_agent_wakeup_distribution():
    # TODO: update this test
    pass


