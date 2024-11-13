import pytest
import logging
import queue
import numpy as np
from datetime import datetime
import logging
from abides_core import Message, NanosecondTime
from typing import Any, Dict, List, Optional, Tuple, Type
from abides_core.agent import Agent
from abides_core.message import Message, MessageBatch, WakeupMsg, SwapMsg, UpdateRateMsg
from abides_core.utils import str_to_ns, merge_swap, fmt_ts
from abides_markets.agents import TradingAgent, LiquidatorAgent
from abides_markets.orders import Side, MarketOrder
from abides_markets.messages.marketdata import MarketDataMsg, L2SubReqMsg
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.messages.order import MarketOrderMsg
from abides_markets.messages.market import MarketClosePriceRequestMsg
logger = logging.getLogger(__name__)
class FakeOrderBook:
    def __init__(self):
        self.twap = 1000
        self.bids = [(1000, 50), (990, 100)]
        self.asks = [(1010, 50), (1020, 100)]
    
    def get_twap(self):
        return self.twap
    
    def set_wakeup(self, agent_id: int, requested_time: NanosecondTime) -> None:
        pass

class FakeKernel:
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
        self.current_time: NanosecondTime = start_time
        self.show_trace_messages: bool = True
        self.messages: queue.PriorityQueue[(int, str, Message)] = queue.PriorityQueue()
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
    def set_wakeup(
        self, sender_id: int, requested_time: Optional[NanosecondTime] = None
    ) -> None:
        """
        Called by an agent to receive a "wakeup call" from the kernel at some requested
        future time.

        NOTE: The agent is responsible for maintaining any required state; the kernel
        will not supply any parameters to the ``wakeup()`` call.

        Arguments:
            sender_id: The ID of the agent making the call.
            requested_time: Defaults to the next possible timestamp.  Wakeup time cannot
            be the current time or a past time.
        """

        if requested_time is None:
            requested_time = self.current_time + 1

        if self.current_time and (requested_time < self.current_time):
            raise ValueError(
                "set_wakeup() called with requested time not in future",
                "current_time:",
                self.current_time,
                "requested_time:",
                requested_time,
            )

        if self.show_trace_messages:
            logger.debug(
                "Kernel adding wakeup for agent {} at time {}".format(
                    sender_id, fmt_ts(requested_time)
                )
            )

        self.messages.put((requested_time, (sender_id, sender_id, WakeupMsg())))


def test_liquidator_agent_initialization():
    logging.debug("Starting test_liquidator_agent_initialization")
    agent_id = 1
    symbol = "PEN"
    wake_up_freq = str_to_ns("1h")
    collateral = 100000
    liquidator = LiquidatorAgent(
        id=agent_id,
        symbol=symbol,
        wake_up_freq=wake_up_freq,
        collateral=collateral,
    )

    # Check if the agent initialization matches the expected values
    logging.debug(f"Agent ID: {liquidator.id}, Expected: {agent_id}")
    assert liquidator.id == agent_id
    
    logging.debug(f"Agent symbol: {liquidator.symbol}, Expected: {symbol}")
    assert liquidator.symbol == symbol

    logging.debug(f"Agent wake_up_freq: {liquidator.wake_up_freq}, Expected: {wake_up_freq}")
    assert liquidator.wake_up_freq == wake_up_freq

    logging.debug(f"Agent collateral: {liquidator.position['COLLATERAL']}, Expected: {collateral}")
    assert liquidator.position["COLLATERAL"] == collateral

    logging.debug(f"Agent state: {liquidator.state}, Expected: AWAITING_WAKEUP")
    assert liquidator.state == "AWAITING_WAKEUP"

    logging.debug(f"Agent watch_list: {liquidator.watch_list}, Expected: []")
    assert liquidator.watch_list == []

    logging.debug(f"Agent failed_liquidation: {liquidator.failed_liquidation}, Expected: 0")
    assert liquidator.failed_liquidation == 0

def test_liquidator_agent_wakeup():
    logging.debug("Starting test_liquidator_agent_wakeup")
    agent_id = 1
    symbol = "PEN"
    wake_up_freq = str_to_ns("1h")
    liquidator = LiquidatorAgent(
        id=agent_id,
        symbol=symbol,
        wake_up_freq=wake_up_freq,
    )
    current_time = 0

    liquidator.kernel = FakeKernel()
    liquidator.exchange_id = liquidator.kernel.exchange_id
    liquidator.mkt_open = True
    liquidator.mkt_closed = False
    liquidator.send_message = lambda recipient_id, message, delay=0: None
    liquidator.get_current_spread = lambda symbol: None

    logging.debug(f"Calling liquidator.wakeup at time: {current_time}")
    liquidator.wakeup(current_time)
    logging.debug(f"Liquidator state after wakeup: {liquidator.state}, Expected: AWAITING_SPREAD")
    assert liquidator.state == "AWAITING_SPREAD"

def test_liquidator_receive_message():
    logging.debug("Starting test_liquidator_receive_message")
    agent_id = 1
    symbol = "PEN"
    liquidator = LiquidatorAgent(
        id=agent_id,
        symbol=symbol,
        subscribe=False,
    )
    current_time = 0
    liquidator.kernel = FakeKernel()
    liquidator.exchange_id = liquidator.kernel.exchange_id
    liquidator.state = "AWAITING_SPREAD"
    liquidator.mkt_open = True
    liquidator.mkt_closed = False

    # Create a watched agent for liquidation
    watched_agent = TradingAgent(id=2)
    watched_agent.position = {"COLLATERAL": 1000, "SIZE": 100, "FIXRATE": 0.05}
    watched_agent.is_healthy = lambda: True
    liquidator.watch_list = [watched_agent]

    liquidator.known_bids = {symbol: [(990, 100)]}
    liquidator.known_asks = {symbol: [(1010, 100)]}

    message = QuerySpreadResponseMsg(
        symbol=symbol,
        bids=liquidator.known_bids[symbol],
        asks=liquidator.known_asks[symbol],
        mkt_closed=False,
        depth=1,
        last_trade=None,
    )

    logging.debug(f"Receiving message for symbol: {symbol} at time: {current_time}")
    liquidator.receive_message(current_time, sender_id=0, message=message)
    logging.debug(f"Failed liquidation after message: {liquidator.failed_liquidation}, Expected: 0")
    assert liquidator.failed_liquidation == 0
    logging.debug(f"Liquidator state after message: {liquidator.state}, Expected: AWAITING_WAKEUP")
    assert liquidator.state == "AWAITING_WAKEUP"

def test_liquidator_check_liquidate_unhealthy_agent():
    logging.debug("Starting test_liquidator_check_liquidate_unhealthy_agent")
    liquidator = LiquidatorAgent(id=1)
    liquidator.kernel = FakeKernel()
    liquidator.exchange_id = liquidator.kernel.exchange_id
    liquidator.symbol = "PEN"
    liquidator.known_bids = {"PEN": [(1000, 50), (990, 100)]}
    liquidator.known_asks = {"PEN": [(1010, 50), (1020, 100)]}

    agent = TradingAgent(id=2)
    agent.kernel = liquidator.kernel
    agent.position = {"COLLATERAL": 10, "SIZE": 100, "FIXRATE": 0.05}
    agent.is_healthy = lambda: False
    agent.mRatio = lambda: 1.2
    agent.mark_to_market = lambda: 50
    agent.cancel_all_orders = lambda: None

    liquidator.place_market_order = lambda symbol, quantity, side: None

    logging.debug(f"Checking liquidate for unhealthy agent with mRatio: {agent.mRatio()}")
    result = liquidator.check_liquidate(agent)
    logging.debug(f"Result of liquidation check: {result}, Expected: True")
    assert result
    logging.debug(f"Agent position after liquidation: {agent.position['SIZE']}, Expected: < 100")
    assert agent.position["SIZE"] < 100
    logging.debug(f"Liquidator position after liquidation: {liquidator.position['SIZE']}, Expected: != 0")
    assert liquidator.position["SIZE"] != 0
    logging.debug(f"Failed liquidation count: {liquidator.failed_liquidation}, Expected: >= 0")
    assert liquidator.failed_liquidation >= 0

def test_liquidator_agent_insufficient_liquidity():
    logging.debug("Starting test_liquidator_agent_insufficient_liquidity")
    liquidator = LiquidatorAgent(id=1)
    liquidator.kernel = FakeKernel()
    liquidator.exchange_id = liquidator.kernel.exchange_id
    liquidator.symbol = "PEN"
    liquidator.known_bids = {"PEN": [(1000, 10)]}
    liquidator.known_asks = {"PEN": [(1010, 10)]}

    agent = TradingAgent(id=2)
    agent.kernel = liquidator.kernel
    agent.position = {"COLLATERAL": 10, "SIZE": 100, "FIXRATE": 0.05}
    agent.is_healthy = lambda: False
    agent.mRatio = lambda: 1.2
    agent.mark_to_market = lambda: 50
    agent.cancel_all_orders = lambda: None

    liquidator.place_market_order = lambda symbol, quantity, side: None

    logging.debug(f"Checking liquidate with insufficient liquidity for agent with mRatio: {agent.mRatio()}")
    result = liquidator.check_liquidate(agent)
    logging.debug(f"Result of liquidation check: {result}, Expected: True")
    assert result
    logging.debug(f"Agent position after liquidation: {agent.position['SIZE']}, Expected: 90")
    assert agent.position["SIZE"] == 90
    logging.debug(f"Failed liquidation count: {liquidator.failed_liquidation}, Expected: >= 0")
    assert liquidator.failed_liquidation >= 0

def test_liquidator_agent_sell_option():
    logging.debug("Starting test_liquidator_agent_sell_option")
    liquidator = LiquidatorAgent(id=1)
    liquidator.kernel = FakeKernel()
    liquidator.exchange_id = liquidator.kernel.exchange_id
    liquidator.symbol = "PEN"
    liquidator.known_bids = {"PEN": [(1000, 50), (990, 100)]}
    liquidator.known_asks = {"PEN": [(1010, 50), (1020, 100)]}

    liquidator.place_market_order = lambda symbol, quantity, side: None

    agent = TradingAgent(id=2)
    agent.kernel = liquidator.kernel
    agent.position = {"COLLATERAL": 10, "SIZE": 100, "FIXRATE": 0.05}
    agent.is_healthy = lambda: False
    agent.mRatio = lambda: 1.2
    agent.mark_to_market = lambda: 50
    agent.cancel_all_orders = lambda: None

    logging.debug(f"Checking liquidate and sell option for agent with mRatio: {agent.mRatio()}")
    result = liquidator.check_liquidate(agent, sell=True)
    logging.debug(f"Result of liquidation check with sell: {result}")
    assert result

def test_liquidator_agent_no_sell_option():
    logging.debug("Starting test_liquidator_agent_no_sell_option")
    liquidator = LiquidatorAgent(id=1)
    liquidator.kernel = FakeKernel()
    liquidator.exchange_id = liquidator.kernel.exchange_id
    liquidator.symbol = "PEN"
    liquidator.known_bids = {"PEN": [(1000, 50), (990, 100)]}
    liquidator.known_asks = {"PEN": [(1010, 50), (1020, 100)]}

    liquidator.place_market_order = lambda symbol, quantity, side: None

    agent = TradingAgent(id=2)
    agent.kernel = liquidator.kernel
    agent.position = {"COLLATERAL": 10, "SIZE": 100, "FIXRATE": 0.05}
    agent.is_healthy = lambda: False
    agent.mRatio = lambda: 1.2
    agent.mark_to_market = lambda: 50
    agent.cancel_all_orders = lambda: None

    logging.debug(f"Checking liquidate without sell option for agent with mRatio: {agent.mRatio()}")
    result = liquidator.check_liquidate(agent, sell=False)
    logging.debug(f"Result of liquidation check without sell: {result}")
    assert result

def test_liquidator_agent_full_integration():
    logging.debug("Starting test_liquidator_agent_full_integration")
    liquidator = LiquidatorAgent(id=1)
    liquidator.kernel = FakeKernel()
    liquidator.exchange_id = liquidator.kernel.exchange_id
    liquidator.symbol = "PEN"
    liquidator.known_bids = {"PEN": [(1000, 50), (990, 100)]}
    liquidator.known_asks = {"PEN": [(1010, 50), (1020, 100)]}

    healthy_agent = TradingAgent(id=2)
    healthy_agent.kernel = liquidator.kernel
    healthy_agent.position = {"COLLATERAL": 1000, "SIZE": 100, "FIXRATE": 0.05}
    healthy_agent.is_healthy = lambda: True

    unhealthy_agent = TradingAgent(id=3)
    unhealthy_agent.kernel = liquidator.kernel
    unhealthy_agent.position = {"COLLATERAL": 10, "SIZE": 100, "FIXRATE": 0.05}
    unhealthy_agent.is_healthy = lambda: False
    unhealthy_agent.mRatio = lambda: 1.2
    unhealthy_agent.mark_to_market = lambda: 50
    unhealthy_agent.cancel_all_orders = lambda: None

    liquidator.place_market_order = lambda symbol, quantity, side: None
    liquidator.mkt_open = True
    liquidator.mkt_closed = False

    current_time = 0
    logging.debug(f"Waking up liquidator at time: {current_time}")
    liquidator.wakeup(current_time)
    message = QuerySpreadResponseMsg(
        symbol=liquidator.symbol,
        bids=liquidator.known_bids[liquidator.symbol],
        asks=liquidator.known_asks[liquidator.symbol],
        mkt_closed=False,
        depth=2,
        last_trade=None,
    )
    logging.debug(f"Receiving spread response at time: {current_time}")
    liquidator.receive_message(current_time, sender_id=0, message=message)

    logging.debug(f"Unhealthy agent position after liquidation: {unhealthy_agent.position['SIZE']}, Expected: < 100")
    assert unhealthy_agent.position["SIZE"] < 100
    logging.debug(f"Liquidator position after liquidation: {liquidator.position['SIZE']}, Expected: != 0")
    assert liquidator.position["SIZE"] != 0
