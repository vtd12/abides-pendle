import logging
import os
import numpy as np
from abides_core.utils import str_to_ns
from abides_core.kernel import Kernel
from abides_markets.agents import ValueAgent, ExchangeAgent
from abides_markets.order_book import OrderBook
from abides_markets.messages.market import MarketHoursMsg
from abides_markets.models import OrderSizeModel
from abides_markets.messages.query import QuerySpreadResponseMsg, QuerySpreadMsg
from abides_markets.orders import Side

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  

from abides_core import Message, NanosecondTime

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

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
    exchange_agent.order_books["PEN"] = OrderBook(owner=exchange_agent, symbol="PEN")

    driving_oracle = FakeDrivingOracle(observed_rates=observed_rates)
    rate_oracle = FakeRateOracle(floating_rate=floating_rate)

    order_size_model = OrderSizeModel()

    value_agent = ValueAgent(
        id=1,
        symbol="PEN",
        random_state=np.random.RandomState(seed=42), 
        collateral=100_000,
        wake_up_freq=str_to_ns("10sec"), 
        r_bar=0.10,
        coef=[0.05, 0.40],
        order_size_model=order_size_model
    )

    kernel = Kernel(
        agents=[exchange_agent, value_agent],
        swap_interval=str_to_ns("1sec"), 
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

def test_value_agent_wakeup_distribution():
    os.makedirs('./logs/', exist_ok=True)

    observed_rates = [1028] * 10
    floating_rate = 0.1095
    mkt_close_time = str_to_ns("100sec") 

    exchange_agent, value_agent, kernel, known_bids, known_asks = setup_agents_and_kernel(
        observed_rates=observed_rates,
        floating_rate=floating_rate,
        expected_sizes=[],
        mkt_close=mkt_close_time,
        exchange_agent_class=ExchangeAgent
    )


    num_samples = 1000
    wake_intervals = []
    for i in range(num_samples):
        interval = value_agent.get_wake_frequency()
        wake_intervals.append(interval)

    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(wake_intervals, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    plt.title('ValueAgent Wakeup Interval Distribution', fontsize=16)
    plt.xlabel('Interval (ns)', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.grid(True)

    mean = np.mean(wake_intervals)
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean:.2f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./logs/value_agent_wakeup_interval_distribution.png', dpi=300)
    plt.close()

    logger.info("Wakeup interval distribution test completed. Check the generated plot.")


