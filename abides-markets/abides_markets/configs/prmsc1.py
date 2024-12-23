# PRMSC-1 (Pendle Reference Market Simulation Configuration):
# - 1       Exchange Agent
# - 10      Noise Agents
# - 5       Value Agents
# - 2       Market Maker Agents
# - 1       Liquidator Agents

import os
from datetime import datetime

import numpy as np
import pandas as pd

from abides_core.utils import get_wake_time, str_to_ns
from abides_markets.agents import (
    ExchangeAgent,
    NoiseAgent,
    ValueAgent,
    PendleSeedingAgent,
    PendleMarketMakerAgent,
    LiquidatorAgent,
)
from abides_markets.models import OrderSizeModel
from abides_markets.utils import generate_latency_model

from abides_markets.rate_oracle import ConstantOracle
from abides_markets.driving_oracle import ManualOracle

########################################################################################################################
############################################### GENERAL CONFIG #########################################################


def build_config(
    seed=int(datetime.now().timestamp() * 1_000_000) % (2**32 - 1),
    start_date="20230101",
    end_date="20230102",
    swap_interval="8h",
    stdout_log_level="INFO",
    ticker="PEN",
    starting_collateral=100,  
    log_orders=True,  # if True log everything
    # 1) Exchange Agent
    book_logging=True,
    book_log_depth=10,
    stream_history_length=500,
    exchange_log_orders=None,
    # 2) Noise Agent
    num_noise_agents=10,
    # 3) Oracle
    kappa = 0.001,  # 0.1% mean reversion
    sigma_s = 100,
    # 4) Value Agents
    num_value_agents=5,
    value_agents_wake_up_freq="10min",
    r_bar=0.10,
    # 5) Market Maker Agents
    num_mm=2,
    mm_agents_wake_up_freq="1h",
    # 6) Liquidator Agents
    num_liq_agents = 1,
    liq_agents_wake_up_freq="10min",
):
    """
    create the background configuration for rmsc04
    These are all the non-learning agent that will run in the simulation
    :param seed: seed of the experiment
    :type seed: int
    :param log_orders: debug mode to print more
    :return: all agents of the config
    :rtype: list
    """

    # fix seed
    np.random.seed(seed)

    # order size model
    ORDER_SIZE_MODEL = OrderSizeModel()  # Order size model

    # date&time
    MKT_OPEN = int(pd.to_datetime(start_date).to_datetime64())
    MKT_CLOSE = int(pd.to_datetime(end_date).to_datetime64())
    SWAP_INT = str_to_ns(swap_interval)

    # driving oracle
    symbols = {
        ticker: {
            "r_bar": r_bar,
            "kappa": kappa,
            "sigma_s": sigma_s
        }
    }

    driving_oracle = ManualOracle(MKT_OPEN, MKT_CLOSE, symbols,
                                  [
                                      {"time": 1/3, "mag": 1000}, 
                                      {"time": 2/3, "mag": -1000}
                                  ]
                                   ) 

    # Agent configuration
    agent_count, agents, agent_types = 0, [], []

    # EXCHANGE
    agents.extend(
        [
            ExchangeAgent(
                id=0,
                name="EXCHANGE_AGENT",
                type="ExchangeAgent",
                mkt_open=MKT_OPEN,
                mkt_close=MKT_CLOSE,
                symbols=[ticker],
                book_logging=book_logging,
                book_log_depth=book_log_depth,
                log_orders=exchange_log_orders,
                pipeline_delay=0,
                computation_delay=0,
                stream_history=stream_history_length,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
            )
        ]
    )
    agent_types.extend("ExchangeAgent")
    agent_count += 1

    # NOISE AGENT
    agents.extend(
        [
            NoiseAgent(
                id=j,
                name="NoiseAgent {}".format(j),
                type="NoiseAgent",
                symbol=ticker,
                collateral=starting_collateral,
                wakeup_time=MKT_OPEN + int((MKT_CLOSE-MKT_OPEN)*np.random.rand()),
                log_orders=log_orders,
                order_size_model=ORDER_SIZE_MODEL,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
            )
            for j in range(agent_count, agent_count + num_noise_agents)
        ]
    )
    agent_count += num_noise_agents
    agent_types.extend(["NoiseAgent"])

    # # FIRST SEEDING AGENT
    # agents.extend(
    #     [
    #         PendleSeedingAgent(
    #             id=j,
    #             name="Seeding Agent {}".format(j),
    #             type="PendleSeedingAgent",
    #             symbol=ticker,
    #             collateral=1000*starting_collateral,
    #             size=seeding_size,
    #             log_orders=log_orders,
    #             random_state=np.random.RandomState(
    #                 seed=np.random.randint(low=0, high=2**32, dtype="uint64")
    #             ),
    #         )
    #         for j in range(agent_count, agent_count + num_seeding_agents)
    #     ]
    # )
    # agent_count += num_seeding_agents
    # agent_types.extend(["PendleSeedingAgent"])

    # VALUE AGENT
    agents.extend(
        [
            ValueAgent(
                id=j,
                name="ValueAgent {}".format(j),
                type="ValueAgent",
                symbol=ticker,
                collateral=starting_collateral,
                log_orders=log_orders,
                order_size_model=ORDER_SIZE_MODEL,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
                r_bar=r_bar,
                wake_up_freq=str_to_ns(value_agents_wake_up_freq)
            )
            for j in range(agent_count, agent_count + num_value_agents)
        ]
    )
    agent_count += num_value_agents
    agent_types.extend(["ValueAgent"])

    # MARKET MAKER AGENT
    agents.extend(
        [
            PendleMarketMakerAgent(
                id=j,
                name="Market Maker Agent {}".format(j),
                type="PendleMarketMakerAgent",
                symbol=ticker,
                collateral=1000*starting_collateral,
                log_orders=log_orders,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
                r_bar=r_bar,
                wake_up_freq=str_to_ns(mm_agents_wake_up_freq)
            )
            for j in range(agent_count, agent_count + num_mm)
        ]
    )
    agent_count += num_mm
    agent_types.extend(["PendleMarketMakerAgent"])

    # LIQUIDATOR
    agents.extend(
        [
            LiquidatorAgent(
                id=j,
                name="Liquidator Agent {}".format(j),
                type="LiquidatorAgent",
                symbol=ticker,
                collateral=1000*starting_collateral,
                log_orders=log_orders,
                random_state=np.random.RandomState(
                    seed=np.random.randint(low=0, high=2**32, dtype="uint64")
                ),
                wake_up_freq=str_to_ns(liq_agents_wake_up_freq)
            )
            for j in range(agent_count, agent_count + num_liq_agents)
        ]
    )
    agent_count += num_liq_agents
    agent_types.extend(["LiquidatorAgent"])


    # extract kernel seed here to reproduce the state of random generator in old version
    random_state_kernel = np.random.RandomState(
        seed=np.random.randint(low=0, high=2**32, dtype="uint64")
    )
    # LATENCY
    latency_model = generate_latency_model(agent_count)

    default_computation_delay = str_to_ns("1s")

    ##kernel args
    kernelStartTime = MKT_OPEN - str_to_ns("1h")
    kernelStopTime = MKT_CLOSE + str_to_ns("1h")
    kernelSwapInt = SWAP_INT

    # PENDLE
    rate_oracle = ConstantOracle(kernelStartTime,
                              kernelStopTime,
                              const_rate = r_bar,
                              starting_market_rate = r_bar
                              )
    # END PENDLE

    return {
        "seed": seed,
        "start_time": kernelStartTime,
        "stop_time": kernelStopTime,
        "swap_interval": kernelSwapInt,
        "agents": agents,
        "agent_latency_model": latency_model,
        "default_computation_delay": default_computation_delay,
        "custom_properties": {"rate_oracle": rate_oracle, 
                              "driving_oracle": driving_oracle},
        "random_state_kernel": random_state_kernel,
        "stdout_log_level": stdout_log_level,
    }
