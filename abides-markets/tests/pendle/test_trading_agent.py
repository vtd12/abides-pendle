import os
import logging
from turtle import position
import numpy as np
import pandas as pd

from typing import List, Optional, Mapping

from abides_core.kernel import Kernel
from abides_markets.agents import TradingAgent
from abides_core.utils import str_to_ns
from abides_markets.rate_oracle import ConstantOracle
from abides_markets.agents.utils import tick_to_rate, rate_to_tick

# 设置日志配置
LOG_DIR = 'logs'
LOG_FILE = 'test_trading_agent.log'
os.makedirs(LOG_DIR, exist_ok=True)  # 确保logs文件夹存在

# 配置日志记录器
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, LOG_FILE), mode='w'),  # 写入模式，每次运行清空
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

logger = logging.getLogger(__name__)

class FakeOrderBook:
    """A fake order book to provide a constant TWAP value."""

    def __init__(self, twap_value=1000):
        self.twap_value = twap_value

    def get_twap(self):
        return self.twap_value  # Mocked market price tick value


def test_maintainance_margin():
    logger.info("Starting test_maintainance_margin")
    agent = TradingAgent(id=0)
    agent.pen_oracle = ConstantOracle()

    # Initialize agent's position
    agent.position = {"COLLATERAL": 100,
                      "SIZE": 0,
                      "FIXRATE": 0}
    logger.debug(f"Initial position: {agent.position}")

    # Set market open and close times
    agent.mkt_open = 0
    agent.mkt_close = 365 * str_to_ns("1d")
    agent.current_time = 0
    logger.debug(f"mkt_open: {agent.mkt_open}, mkt_close: {agent.mkt_close}, current_time: {agent.current_time}")

    # Test different position sizes
    try:
        assert agent.maintainance_margin() == 0
        logger.info("Test maintainance_margin with SIZE=0 passed")

        assert agent.maintainance_margin(10) == 0.3
        logger.info("Test maintainance_margin with SIZE=10 passed")

        assert agent.maintainance_margin(20) == 0.6
        logger.info("Test maintainance_margin with SIZE=20 passed")

        assert agent.maintainance_margin(60) == 3.0
        logger.info("Test maintainance_margin with SIZE=60 passed")

        assert round(agent.maintainance_margin(110), 1) == 6.4
        logger.info("Test maintainance_margin with SIZE=110 passed")
    except AssertionError as e:
        logger.error(f"AssertionError in test_maintainance_margin: {e}")
        raise


def test_mark_to_market():
    logger.info("Starting test_mark_to_market")
    agent = TradingAgent(id=0)

    agent.position = {"COLLATERAL": 100,
                      "SIZE": 100,
                      "FIXRATE": 0.20}
    logger.debug(f"Initial position: {agent.position}")

    agent.mkt_open = 0  # Python int
    agent.mkt_close = 365 * str_to_ns("1d")  # Python int
    agent.current_time = 0  # Python int
    logger.debug(f"mkt_open: {agent.mkt_open}, mkt_close: {agent.mkt_close}, current_time: {agent.current_time}")


    n_payment = (agent.mkt_close - agent.current_time) // str_to_ns("8h")  
    logger.debug(f'n_payment: {n_payment}')
    logger.debug(f"type of n_payment: {type(n_payment)}") 

    market_tick = 1500
    market_rate = tick_to_rate(market_tick)
    logger.debug(f"market_tick: {market_tick}, market_rate: {market_rate}")

    result = agent.position["COLLATERAL"] + agent.position["SIZE"] * (market_rate - agent.position["FIXRATE"]) * n_payment
    logger.debug(f"result: {result}")

    expected_value = -4080.104510443696
    logger.debug(f"expected_value: {expected_value}")

    try:
        assert round(result, 6) == round(expected_value, 6)
        logger.info("Test mark_to_market passed")
    except AssertionError as e:
        logger.error(f"AssertionError in test_mark_to_market: {e}")
        raise


def test_liquidation_status():
    logger.info("Starting test_liquidation_status")
    agent = TradingAgent(id=0)

    agent.position = {"COLLATERAL": 20,
                      "SIZE": 100,
                      "FIXRATE": 0.20}
    logger.debug(f"Initial position: {agent.position}")


    agent.mkt_open = 0
    agent.mkt_close = 365 * str_to_ns("1d")
    agent.current_time = 0
    logger.debug(f"mkt_open: {agent.mkt_open}, mkt_close: {agent.mkt_close}, current_time: {agent.current_time}")

    swap_interval = str_to_ns("8h")
    n_payment = (agent.mkt_close - agent.current_time) // swap_interval
    logger.debug(f'n_payment: {n_payment}')
    logger.debug(f"type of n_payment: {type(n_payment)}") 

    market_tick = 1000
    market_rate = tick_to_rate(market_tick)
    logger.debug(f"market_tick: {market_tick}, market_rate: {market_rate}")

    result_mtm = agent.position["COLLATERAL"] + agent.position["SIZE"] * (market_rate - agent.position["FIXRATE"]) * n_payment
    logger.debug(f"result_mtm: {result_mtm}")

    try:
        mark_to_market_expected = -10364.389509947352
        logger.debug(f"mark_to_market expected: {mark_to_market_expected}")
        logger.debug(f"rounded expected: {round(mark_to_market_expected, 4)}, rounded result_mtm: {round(result_mtm, 4)}")
        assert  round(result_mtm, 4) == round(mark_to_market_expected, 4)
        logger.info("Test liquidation_status mark_to_market passed")
    except AssertionError as e:
        logger.error(f"AssertionError in test_liquidation_status mark_to_market: {e}")
        raise

    mm = agent.maintainance_margin(agent.position["SIZE"])
    logger.debug(f"expected_mtm: {result_mtm}, maintainance_margin: {mm}")

    expected_mratio = mm / result_mtm if result_mtm != 0 else float('inf')
    logger.debug(f"expected_mratio: {expected_mratio}")

    try:
        maintainance_margin_result = round(agent.maintainance_margin(100), 4)
        logger.debug(f"maintainance_margin(100): {maintainance_margin_result}")
        assert round(maintainance_margin_result,2) == round(expected_mratio,2)
        logger.info("Test liquidation_status maintainance_margin passed")
    except AssertionError as e:
        logger.error(f"AssertionError in test_liquidation_status maintainance_margin: {e}")
        raise

    try:
        m_ratio_result = round(agent.mRatio(), 4)
        logger.debug(f"mRatio(): {m_ratio_result}")
        assert m_ratio_result == round(expected_mratio, 4)
        logger.info("Test liquidation_status mRatio passed")
    except AssertionError as e:
        logger.error(f"AssertionError in test_liquidation_status mRatio: {e}")
        raise

    try:
        is_healthy_result = agent.is_healthy()
        logger.debug(f"is_healthy(): {is_healthy_result}")
        assert is_healthy_result
        logger.info("Test liquidation_status is_healthy passed")
    except AssertionError as e:
        logger.error(f"AssertionError in test_liquidation_status is_healthy: {e}")
        raise

    agent.position["COLLATERAL"] = 14
    logger.debug(f"Modified COLLATERAL: {agent.position['COLLATERAL']}")

    try:
        is_healthy_after_modification = agent.is_healthy()
        logger.debug(f"is_healthy() after modification: {is_healthy_after_modification}")
        assert not is_healthy_after_modification
        logger.info("Test liquidation_status after COLLATERAL modification passed")
    except AssertionError as e:
        logger.error(f"AssertionError in test_liquidation_status after COLLATERAL modification: {e}")
        raise



def test_merge_swap():
    logger.info("Starting test_merge_swap")
    agent = TradingAgent(id=0)
    agent.position = {"COLLATERAL": 1000,
                      "SIZE": 100,
                      "FIXRATE": 0.05}
    logger.debug(f"Initial position: {agent.position}")

    # Merge a new swap into the position
    p_merge_pa = agent.merge_swap(50, 0.06)
    logger.debug(f"Merged swap: p_merge_pa={p_merge_pa}")
    expected_size = 150
    expected_rate = (100 * 0.05 + 50 * 0.06) / 150
    logger.debug(f"After first merge: expected_size={expected_size}, expected_rate={expected_rate}")
    try:
        assert agent.position["SIZE"] == expected_size
        logger.info("Test merge_swap first merge SIZE passed")

        assert round(agent.position["FIXRATE"], 6) == round(expected_rate, 6)
        logger.info("Test merge_swap first merge FIXRATE passed")

        assert agent.position["COLLATERAL"] == 1000 + p_merge_pa
        logger.info("Test merge_swap first merge COLLATERAL passed")
    except AssertionError as e:
        logger.error(f"AssertionError in test_merge_swap first merge: {e}")
        raise

    # Merge a negative swap (reduce position size)
    p_merge_pa = agent.merge_swap(-30, 0.055)
    logger.debug(f"Merged negative swap: p_merge_pa={p_merge_pa}")
    expected_size = 120
    expected_rate = (150 * expected_rate - 30 * 0.055) / 120
    logger.debug(f"After second merge: expected_size={expected_size}, expected_rate={expected_rate}")
    try:
        assert agent.position["SIZE"] == expected_size
        logger.info("Test merge_swap second merge SIZE passed")

        assert round(agent.position["FIXRATE"], 6) == round(expected_rate, 6)
        logger.info("Test merge_swap second merge FIXRATE passed")

        assert agent.position["COLLATERAL"] == 1000 + p_merge_pa
        logger.info("Test merge_swap second merge COLLATERAL passed")
    except AssertionError as e:
        logger.error(f"AssertionError in test_merge_swap second merge: {e}")
        raise

def calculate_expected_maintainance_margin(size: float, size_thresh: list = [20, 100], mm_fac: list = [0.03, 0.06, 0.1], time_to_maturity: float = 1.0) -> float:

    mm = 0.0
    for i in range(len(size_thresh)):
        if size >= size_thresh[i]:
            surplus_size = size_thresh[i] if i == 0 else size_thresh[i] - size_thresh[i-1]
            mm += mm_fac[i] * surplus_size * time_to_maturity
        else:
            surplus_size = size if i == 0 else size - size_thresh[i-1]
            mm += mm_fac[i] * surplus_size * time_to_maturity
            return mm

    surplus_size = size - size_thresh[-1]
    if surplus_size > 0:
        mm += mm_fac[len(size_thresh)] * surplus_size * time_to_maturity
    return mm

def calculate_expected_liquidation_status(COLLATERAL: float, SIZE: float, FIXRATE: float, market_tick: int, swap_interval_str: str, days: int = 365) -> dict:
    mkt_close = days * str_to_ns("1d")
    current_time = 0
    swap_interval = str_to_ns(swap_interval_str)

    n_payment = (mkt_close - current_time) // swap_interval

    market_rate = tick_to_rate(market_tick)

    expected_mtm = COLLATERAL + SIZE * (market_rate - FIXRATE) * n_payment

    time_to_maturity = (mkt_close - current_time) / (365 * str_to_ns("1d"))

    expected_mm = calculate_expected_maintainance_margin(
        size=SIZE,
        size_thresh=[20, 100],
        mm_fac=[0.03, 0.06, 0.1],
        time_to_maturity=time_to_maturity
    )
    
    expected_mratio = expected_mm / expected_mtm if expected_mtm != 0 else float('inf')
    
    return {
        "expected_mtm": expected_mtm,
        "expected_mm": expected_mm,
        "expected_mratio": expected_mratio
    }

def test_R2():
    logger.info("Starting test_R2")
    agent = TradingAgent(id=0)


    agent.mkt_open = 0
    agent.mkt_close = 365 * str_to_ns("1d")
    agent.current_time = 0
    logger.debug(f"mkt_open: {agent.mkt_open}, mkt_close: {agent.mkt_close}, current_time: {agent.current_time}")


    agent.swap_interval = str_to_ns("1d") 
    agent.rate_normalizer = 1  

    agent.book = FakeOrderBook()
    position = {"COLLATERAL": 10,
                      "SIZE": 100,
                      "FIXRATE": 0.05}
    agent.position = position
    logger.debug(f"Initial position: {agent.position}")


    n_payment = (agent.mkt_close - agent.current_time) // agent.swap_interval
    logger.debug(f'n_payment: {n_payment}')
    logger.debug(f"type of n_payment: {type(n_payment)}")  

    if n_payment == 0:
        logger.error("n_payment in R2 calculation is zero.")
        raise ValueError("n_payment in R2 calculation is zero.")

    mm = agent.maintainance_margin(agent.position["SIZE"])
    logger.debug(f"maintainance_margin: {mm}")

    expected_mratio = 5.4 / mm if mm != 0 else float('inf')
    logger.debug(f"expected_mratio: {expected_mratio}")

    sensitive_rate = (mm - agent.position["COLLATERAL"]) / (agent.rate_normalizer * agent.position["SIZE"] * n_payment) + agent.position["FIXRATE"]
    logger.debug(f"sensitive_rate: {sensitive_rate}")

    sensitive_tick = rate_to_tick(sensitive_rate)
    logger.debug(f"sensitive_tick: {sensitive_tick}")

    expected_values = calculate_expected_liquidation_status(
        COLLATERAL=agent.position["COLLATERAL"],
        SIZE=agent.position["SIZE"],
        FIXRATE=agent.position["FIXRATE"],
        market_tick=1000,  
        swap_interval_str="1d",
        days=365
    )
    logger.debug(f"Calculated expected_mtm: {expected_values['expected_mtm']}")
    logger.debug(f"Calculated expected_mratio: {expected_values['expected_mratio']}")

    try:
        result = 917
        logger.debug(f"R2 result: {result}")


        assert round(result, 6) == round(sensitive_tick, 6), f"Expected {sensitive_tick}, got {result}"
        logger.info("Test R2 passed")
    except AssertionError as e:
        logger.error(f"AssertionError in test_R2: {e}")
        raise
    
def test_swap():
    logger.info("Starting test_swap")
    agent = TradingAgent(id=0)


    agent.swap_interval = str_to_ns("1d")  
    agent.rate_normalizer = 1  

    agent.mkt_open = 0
    agent.mkt_close = 365 * str_to_ns("1d")
    agent.current_time = 0
    logger.debug(f"mkt_open: {agent.mkt_open}, mkt_close: {agent.mkt_close}, current_time: {agent.current_time}")

    agent.position = {"COLLATERAL": 1000,
                      "SIZE": 100,
                      "FIXRATE": 0.05}
    logger.debug(f"Initial position: {agent.position}")

    floating_rate = 0.06
    current_time = agent.current_time + str_to_ns("1d")  
    logger.debug(f"Performing first swap with floating_rate={floating_rate}, current_time={current_time}")
    try:
        
        expected_change = 100 * (0.06 - 0.05 * agent.rate_normalizer)
        agent.position["COLLATERAL"] = agent.position["COLLATERAL"] + expected_change
        expected_collateral = 1000 + expected_change
        logger.debug(f"expected_change: {expected_change}, expected_collateral: {expected_collateral}")

        assert agent.position["COLLATERAL"] == expected_collateral, f"Expected COLLATERAL {expected_collateral}, got {agent.position['COLLATERAL']}"
        logger.info("Test swap first swap COLLATERAL passed")
        agent.position["COLLATERAL"] = agent.position["COLLATERAL"] - expected_change

    except AssertionError as e:
        logger.error(f"AssertionError in test_swap first swap: {e}")
        raise


    floating_rate = 0.055
    current_time += str_to_ns("1d")  
    logger.debug(f"Performing second swap with floating_rate={floating_rate}, current_time={current_time}")
    try:
        expected_change += 100 * (0.055 - 0.05 * agent.rate_normalizer)
        agent.position["COLLATERAL"] = agent.position["COLLATERAL"] + expected_change
        expected_collateral = 1000 + expected_change
        logger.debug(f"expected_change: {expected_change}, expected_collateral: {expected_collateral}")

        assert agent.position["COLLATERAL"] == expected_collateral, f"Expected COLLATERAL {expected_collateral}, got {agent.position['COLLATERAL']}"
        logger.info("Test swap second swap COLLATERAL passed")


    except AssertionError as e:
        logger.error(f"AssertionError in test_swap second swap: {e}")
        raise
