import logging
from datetime import datetime
from matplotlib.pylab import f
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Type
from abides_core.agent import Agent
from abides_core.message import Message, MessageBatch, WakeupMsg, SwapMsg, UpdateRateMsg
from abides_markets.agents import TradingAgent
from abides_core.utils import str_to_ns, fmt_ts
from abides_markets.rate_oracle import ConstantOracle
from abides_markets.agents.utils import tick_to_rate, rate_to_tick

logger = logging.getLogger(__name__)


class FakeOrderBook:
    """A fake order book to provide a constant TWAP value."""

    def __init__(self, twap_value=1000):
        self.twap_value = twap_value

    def get_twap(self):
        return self.twap_value  # Mocked market price tick value

class Kernel:
    def __init__(self,
        agents=[],
        # PENDLE
        swap_interval = str_to_ns("8h"),
        # END PENDLE
        start_time= str_to_ns("00:00:00"),
        stop_time= str_to_ns("23:00:00"),
        default_computation_delay: int = 1,
        default_latency: float = 1,
        agent_latency: Optional[List[List[float]]] = None,
        skip_log: bool = True,
        seed: Optional[int] = None,
        log_dir: Optional[str] = None,
        custom_properties: Optional[Dict[str, Any]] = None,
        random_state: Optional[np.random.RandomState] = None,):
        
        self.book = FakeOrderBook()
        self.rate_normalizer = 1
        self.swap_interval = 1
        self.exchange_id = 0
        logger.debug(f"Kernel initialized")
    
    def run(self):
        self.initialize()

        self.runner()

        return self.terminate()
    
    def initialize(self):
        logger.info(f"Simulation started at {fmt_ts(self.current_time)}!")
        logger.debug("--- Agent.kernel_initializing() ---")
        for agent in self.agents:
            agent.kernel_initializing(self)
        logger.debug("--- Agent.kernel_starting() ---")
        for agent in self.agents:
            agent.kernel_starting(self.start_time)

        # Set the kernel to its start_time.
        self.current_time = self.start_time

        logger.debug("--- Kernel Clock started ---")
        logger.debug("Kernel.current_time is now {}".format(fmt_ts(self.current_time)))

        # Start processing the Event Queue.
        logger.debug("--- Kernel Event Queue begins ---")
        logger.debug(
            "Kernel will start processing messages. Queue length: {}".format(
                len(self.messages.queue)
            )
        )

        # Track starting wall clock time and total message count for stats at the end.
        self.event_queue_wall_clock_start = datetime.now()
        self.ttl_messages = 0

        # PENDLE: 
        # Push message of swaps into message list
        mkt_open, mkt_close = self.agents[0].mkt_open, self.agents[0].mkt_close
        swap_time = mkt_open + self.swap_interval

        while swap_time <= mkt_close:
            for agent in self.agents[1:]:  # Only swap with trading agents
                self.messages.put((swap_time, (-1, agent.id, SwapMsg())))
            swap_time += self.swap_interval

        # END PENDLE
    def send_message(self, sender_id, recipient_id, message, delay=0):
        pass
    
    def runner(
        self, agent_actions: Optional[Tuple[Agent, List[Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        """
        Start the simulation and processing of the message queue.
        Possibility to add the optional argument agent_actions. It is a list of dictionaries corresponding
        to actions to be performed by the experimental agent (Gym Agent).

        Arguments:
            agent_actions: A list of the different actions to be performed represented in a dictionary per action.

        Returns:
          - it is a dictionnary composed of two elements:
            - "done": boolean True if the simulation is done, else False. It is true when simulation reaches end_time or when the message queue is empty.
            - "results": it is the raw_state returned by the gym experimental agent, contains data that will be formated in the gym environement to formulate state, reward, info etc.. If
               there is no gym experimental agent, then it is None.
        """
        # run an action on a given agent before resuming queue: to be used to take exp agent action before resuming run
        if agent_actions is not None:
            exp_agent, action_list = agent_actions
            exp_agent.apply_actions(action_list)

        # Process messages until there aren't any (at which point there never can
        # be again, because agents only "wake" in response to messages), or until
        # the kernel stop time is reached.
        while (
            not self.messages.empty()
            and self.current_time
            and (self.current_time <= self.stop_time)
        ):
            # Get the next message in timestamp order (delivery time) and extract it.
            self.current_time, event = self.messages.get()
            assert self.current_time is not None

            sender_id, recipient_id, message = event

            # Periodically print the simulation time and total messages, even if muted.
            if self.ttl_messages % 1_000_000 == 0:
                logger.info(
                    "--- Simulation time: {}, messages processed: {:,}, wallclock elapsed: {:.2f}s ---".format(
                        fmt_ts(self.current_time),
                        self.ttl_messages,
                        (
                            datetime.now() - self.event_queue_wall_clock_start
                        ).total_seconds(),
                    )
                )

            if self.show_trace_messages:
                logger.debug("--- Kernel Event Queue pop ---")
                logger.debug(
                    "Kernel handling {} message for agent {} at time {}".format(
                        message.type(), recipient_id, self.current_time
                    )
                )

            self.ttl_messages += 1

            # In between messages, always reset the current_agent_additional_delay.
            self.current_agent_additional_delay = 0

            # Dispatch message to agent.
            if isinstance(message, WakeupMsg):
                # Test to see if the agent is already in the future.  If so,
                # delay the wakeup until the agent can act again.
                if self.agent_current_times[recipient_id] > self.current_time:
                    # Push the wakeup call back into the PQ with a new time.
                    self.messages.put(
                        (
                            self.agent_current_times[recipient_id],
                            (sender_id, recipient_id, message),
                        )
                    )
                    if self.show_trace_messages:
                        logger.debug(
                            "After wakeup return, agent {} delayed from {} to {}".format(
                                recipient_id,
                                fmt_ts(self.current_time),
                                fmt_ts(self.agent_current_times[recipient_id]),
                            )
                        )
                    continue

                # Set agent's current time to global current time for start
                # of processing.
                self.agent_current_times[recipient_id] = self.current_time

                # Wake the agent and get value passed to kernel to listen for kernel interruption signal
                wakeup_result = self.agents[recipient_id].wakeup(self.current_time)

                # Delay the agent by its computation delay plus any transient additional delay requested.
                self.agent_current_times[recipient_id] += (
                    self.agent_computation_delays[recipient_id]
                    + self.current_agent_additional_delay
                )

                if self.show_trace_messages:
                    logger.debug(
                        "After wakeup return, agent {} delayed from {} to {}".format(
                            recipient_id,
                            fmt_ts(self.current_time),
                            fmt_ts(self.agent_current_times[recipient_id]),
                        )
                    )
                # catch kernel interruption signal and return wakeup_result which is the raw state from gym agent
                if wakeup_result != None:
                    return {"done": False, "result": wakeup_result}

            # PENDLE: Detect Swap Msg
            elif isinstance(message, SwapMsg):
                # Test to see if the agent is already in the future.  If so,
                # delay the swap until the agent can act again.
                if self.agent_current_times[recipient_id] > self.current_time:
                    # Push the wakeup call back into the PQ with a new time.
                    self.messages.put(
                        (
                            self.agent_current_times[recipient_id],
                            (sender_id, recipient_id, message),
                        )
                    )
                    if self.show_trace_messages:
                        logger.debug(
                            "After swap return, agent {} delayed from {} to {}".format(
                                recipient_id,
                                fmt_ts(self.current_time),
                                fmt_ts(self.agent_current_times[recipient_id]),
                            )
                        )
                    continue

                # Set agent's current time to global current time for start
                # of processing.
                self.agent_current_times[recipient_id] = self.current_time

                # Swap the agent and get value passed to kernel to listen for kernel interruption signal
                swap_result = self.agents[recipient_id].swap(self.current_time, self.rate_oracle.get_floating_rate(self.current_time))

                # Delay the agent by its computation delay plus any transient additional delay requested.
                self.agent_current_times[recipient_id] += (
                    self.agent_computation_delays[recipient_id]
                    + self.current_agent_additional_delay
                )

                if self.show_trace_messages:
                    logger.debug(
                        "After swap return, agent {} delayed from {} to {}".format(
                            recipient_id,
                            fmt_ts(self.current_time),
                            fmt_ts(self.agent_current_times[recipient_id]),
                        )
                    )
                # catch kernel interruption signal and return swap_result
                if swap_result != None:
                    return {"done": False, "result": swap_result}
                
            # END PENDLE
            else:
                # Test to see if the agent is already in the future.  If so,
                # delay the message until the agent can act again.
                if self.agent_current_times[recipient_id] > self.current_time:
                    # Push the message back into the PQ with a new time.
                    self.messages.put(
                        (
                            self.agent_current_times[recipient_id],
                            (sender_id, recipient_id, message),
                        )
                    )
                    if self.show_trace_messages:
                        logger.debug(
                            "Agent in future: message requeued for {}".format(
                                fmt_ts(self.agent_current_times[recipient_id])
                            )
                        )
                    continue

                # Set agent's current time to global current time for start
                # of processing.
                self.agent_current_times[recipient_id] = self.current_time

                # Deliver the message.
                if isinstance(message, MessageBatch):
                    messages = message.messages
                else:
                    messages = [message]

                for message in messages:
                    # Delay the agent by its computation delay plus any transient additional delay requested.
                    self.agent_current_times[recipient_id] += (
                        self.agent_computation_delays[recipient_id]
                        + self.current_agent_additional_delay
                    )

                    if self.show_trace_messages:
                        logger.debug(
                            "After receive_message return, agent {} delayed from {} to {}".format(
                                recipient_id,
                                fmt_ts(self.current_time),
                                fmt_ts(self.agent_current_times[recipient_id]),
                            )
                        )

                    self.agents[recipient_id].receive_message(
                        self.current_time, sender_id, message
                    )

        if self.messages.empty():
            logger.info("--- Kernel Event Queue empty ---")

        if self.current_time and (self.current_time > self.stop_time):
            logger.info(f"--- Kernel Stop Time {self.stop_time} surpassed ---")

        return {"done": True, "result": None}


def test_maintainance_margin():
    logging.debug("Starting test_maintainance_margin")
    agent = TradingAgent(id=0)
    agent.pen_oracle = ConstantOracle()

    # Initialize agent's position
    agent.position = {"COLLATERAL": 100, "SIZE": 0, "FIXRATE": 0}
    agent.mkt_open = 0
    agent.mkt_close = 365 * str_to_ns("1d")  # 365 days in nanoseconds
    agent.current_time = 0

    # Test different position sizes
    for size, expected_margin in [(0, 0), (10, 0.3), (20, 0.6), (60, 3.0), (110, 6.4)]:
        margin = agent.maintainance_margin(size)
        logging.debug(f"Position size: {size}, Expected margin: {expected_margin}, Calculated margin: {margin}")
        assert round(margin, 1) == expected_margin

def test_mark_to_market():
    logging.debug("Starting test_mark_to_market")
    agent = TradingAgent(id=0)
    kernel = Kernel([agent], swap_interval=str_to_ns("8h"))
    kernel.book = FakeOrderBook()
    kernel.rate_normalizer = 1
    agent.kernel = kernel

    # Initialize agent's position
    agent.position = {"COLLATERAL": 100, "SIZE": 100, "FIXRATE": 0.20}
    agent.mkt_open = 0
    agent.mkt_close = 365 * str_to_ns("1d")
    agent.current_time = 0

    n_payment = int((agent.mkt_close - agent.current_time) // agent.kernel.swap_interval)
    for market_tick in [1500, agent.kernel.book.get_twap()]:
        market_rate = tick_to_rate(market_tick)
        expected_value = agent.position["COLLATERAL"] + agent.position["SIZE"] * (market_rate - agent.position["FIXRATE"]) * n_payment
        result = agent.mark_to_market(market_tick=market_tick, log=False)
        logging.debug(f"Market tick: {market_tick}, Market rate: {market_rate}, Expected MTM value: {expected_value}, Calculated MTM value: {result}")
        assert round(result, 6) == round(expected_value, 6)

def test_liquidation_status():
    logging.debug("Starting test_liquidation_status")
    agent = TradingAgent(id=0)
    kernel = Kernel([agent], swap_interval=str_to_ns("8h"))
    kernel.book = FakeOrderBook()
    kernel.rate_normalizer = 1
    agent.kernel = kernel

    agent.position = {"COLLATERAL": 20, "SIZE": 100, "FIXRATE": 0.20}
    agent.mkt_open = 0
    agent.mkt_close = 365 * str_to_ns("1d")
    agent.current_time = 0

    n_payment = int((agent.mkt_close - agent.current_time) // agent.kernel.swap_interval)
    market_tick = agent.kernel.book.get_twap()
    market_rate = tick_to_rate(market_tick)
    expected_mtm = agent.position["COLLATERAL"] + agent.position["SIZE"] * (market_rate - agent.position["FIXRATE"]) * n_payment
    result = agent.mark_to_market(log=False)
    logging.debug(f"Market tick: {market_tick}, Market rate: {market_rate}, Expected MTM: {expected_mtm}, Calculated MTM: {result}")
    assert round(result, 4) == round(expected_mtm, 4)

    expected_margin = 5.4
    calculated_margin = agent.maintainance_margin(100)
    logging.debug(f"Expected maintenance margin: {expected_margin}, Calculated margin: {calculated_margin}")
    assert round(calculated_margin, 4) == expected_margin

    expected_mratio = expected_margin / expected_mtm
    mratio = agent.mRatio()
    logging.debug(f"Expected M-ratio: {expected_mratio}, Calculated M-ratio: {mratio}")
    assert round(mratio, 4) == round(expected_mratio, 4)

    assert agent.is_healthy()
    agent.position["COLLATERAL"] = 14
    logging.debug(f"Adjusted COLLATERAL to 14, Checking health status: {agent.is_healthy()}")
    assert not agent.is_healthy()

def test_merge_swap():
    logging.debug("Starting test_merge_swap")
    agent = TradingAgent(id=0)
    agent.position = {"COLLATERAL": 1000, "SIZE": 100, "FIXRATE": 0.05}

    # Test merge with positive swap
    p_merge_pa = agent.merge_swap(50, 0.06)
    expected_size = 150
    expected_rate = (100 * 0.05 + 50 * 0.06) / 150
    logging.debug(f"New size after merge: {agent.position['SIZE']}, Expected size: {expected_size}, Calculated FIXRATE: {agent.position['FIXRATE']}, Expected FIXRATE: {expected_rate}")
    assert agent.position["SIZE"] == expected_size
    assert round(agent.position["FIXRATE"], 6) == round(expected_rate, 6)
    assert agent.position["COLLATERAL"] == 1000 + p_merge_pa

    # Test merge with negative swap
    p_merge_pa = agent.merge_swap(-30, 0.055)
    expected_size = 120
    expected_rate = (150 * expected_rate - 30 * 0.055) / 120
    logging.debug(f"New size after merge: {agent.position['SIZE']}, Expected size: {expected_size}, Calculated FIXRATE: {agent.position['FIXRATE']}, Expected FIXRATE: {expected_rate}")
    assert agent.position["SIZE"] == expected_size
    assert round(agent.position["FIXRATE"], 6) == round(expected_rate, 6)
    assert agent.position["COLLATERAL"] == 1000 + p_merge_pa

def test_R2():
    logging.debug("Starting test_R2")
    agent = TradingAgent(id=0)
    agent.mkt_open = 0
    agent.mkt_close = 365 * str_to_ns("1d")
    agent.current_time = 0
    kernel = Kernel([agent], swap_interval=str_to_ns("1d"))
    kernel.rate_normalizer = 1
    agent.kernel = kernel

    agent.position = {"COLLATERAL": 10, "SIZE": 100, "FIXRATE": 0.05}
    n_payment = int((agent.mkt_close - agent.current_time) // agent.kernel.swap_interval)
    mm = agent.maintainance_margin(agent.position["SIZE"])
    sensitive_rate = (mm - agent.position["COLLATERAL"]) / (agent.kernel.rate_normalizer * agent.position["SIZE"] * n_payment) + agent.position["FIXRATE"]
    sensitive_tick = rate_to_tick(sensitive_rate)
    result = agent.R2()
    logging.debug(f"Expected R2 tick: {sensitive_tick}, Calculated R2 tick: {result}")
    assert round(result, 6) == round(sensitive_tick, 6)

def test_mRatio_and_is_healthy():
    logging.debug("Starting test_mRatio_and_is_healthy")
    agent = TradingAgent(id=0)
    kernel = Kernel([agent], swap_interval=str_to_ns("1d"))
    kernel.rate_normalizer = 1
    agent.kernel = kernel
    agent.mkt_open = 0
    agent.mkt_close = 365 * str_to_ns("1d")
    agent.current_time = 0

    # Test with healthy position
    agent.position = {"COLLATERAL": 50, "SIZE": 100, "FIXRATE": 0.05}
    m_ratio = agent.mRatio()
    logging.debug(f"Calculated M-ratio (should be < 1): {m_ratio}")
    assert m_ratio < 1
    assert agent.is_healthy()

    # Test with unhealthy position
    agent.position = {"COLLATERAL": 1, "SIZE": 100, "FIXRATE": 0.05}
    m_ratio = agent.mRatio()
    logging.debug(f"Calculated M-ratio (should be >= 1): {m_ratio}")
    assert m_ratio >= 1
    assert not agent.is_healthy()

def test_swap():
    logging.debug("Starting test_swap")
    agent = TradingAgent(id=0)
    kernel = Kernel([agent], swap_interval=str_to_ns("1d"))
    kernel.rate_normalizer = 1
    agent.kernel = kernel
    agent.mkt_open = 0
    agent.mkt_close = 365 * str_to_ns("1d")
    agent.current_time = 0

    agent.position = {"COLLATERAL": 1000, "SIZE": 100, "FIXRATE": 0.05}

    # Test swap with floating rate
    floating_rate = 0.06
    current_time = agent.current_time + str_to_ns("1d")
    agent.swap(current_time=current_time, floating_rate=floating_rate)

    expected_change = 100 * (0.06 - 0.05 * kernel.rate_normalizer)
    expected_collateral = 1000 + expected_change
    logging.debug(f"Floating rate: {floating_rate}, Expected COLLATERAL: {expected_collateral}, Calculated COLLATERAL: {agent.position['COLLATERAL']}")
    assert agent.position["COLLATERAL"] == expected_collateral
    assert agent.current_time == current_time