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
        self.last_twap = -3.230998
    
    def get_twap(self):
        return self.last_twap
    
    def set_wakeup(self, agent_id: int, requested_time: NanosecondTime) -> None:
        pass
class FakeOracle:
    def __init__(self):
        pass


class FakeKernel:
    def __init__(self,
        agents = {},
        # PENDLE
        swap_interval = str_to_ns("8h"),
        # END PENDLE
        start_time= str_to_ns("00:00:00"),
        ):
        self.agents = {agent.id: agent for agent in agents}
        self.current_time: NanosecondTime = start_time
        self.show_trace_messages: bool = True
        self.messages: queue.PriorityQueue[(int, str, Message)] = queue.PriorityQueue()
        self.book = FakeOrderBook()
        self.rate_normalizer = 1
        self.swap_interval = str_to_ns("8h")
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


def test_liquidator_calculate():
    """
    Test the full liquidation process, ensuring correct position and collateral adjustment.
    This verifies the liquidation of a user's position and the correct transfer of collateral.
    """
    logger.debug("Starting test_liquidator_calculate_liquidation")
    
    # Initialize agents
    logger.debug("Initializing LiquidatorAgent and TradingAgent")
    liquidator = LiquidatorAgent(id=0)
    unhealthy_agent = TradingAgent(id=1)
    
    # Create and configure the kernel
    logger.debug("Creating and configuring FakeKernel")
    kernel = FakeKernel(agents=[liquidator, unhealthy_agent], swap_interval=str_to_ns("8h"))
    kernel.book = FakeOrderBook()
    
    # Assign the kernel to agents
    logger.debug("Assigning kernel to agents")
    liquidator.kernel = kernel
    unhealthy_agent.kernel = kernel
    
    # Set up the liquidator agent
    logger.debug("Setting up LiquidatorAgent")
    liquidator.exchange_id = kernel.exchange_id
    liquidator.symbol = "PEN"
    liquidator.known_bids = {"PEN": [(-3.230998, 5000), (-3.4, 10000)]}
    liquidator.known_asks = {"PEN": [(-3.230998, 5000), (-2.9, 10000)]}
    liquidator.mkt_open = True
    liquidator.mkt_closed = False
    # Mock the place_market_order method if needed
    liquidator.place_market_order = lambda symbol, quantity, side: logger.debug(
        f"Liquidator placed market order: symbol={symbol}, quantity={quantity}, side={side}"
    )
    
    # Set up the unhealthy agent
    logger.debug("Setting up unhealthy TradingAgent")
    unhealthy_agent.position = {"COLLATERAL": 50, "SIZE": 100, "FIXRATE": 0.0001}
    unhealthy_agent.mkt_open = 0
    unhealthy_agent.mkt_close = 365 * str_to_ns("1d")
    unhealthy_agent.current_time = 0
    
    # Calculate and log initial values
    logger.debug("Calculating initial values for unhealthy agent")
    maintenance_margin = unhealthy_agent.maintainance_margin(unhealthy_agent.position["SIZE"])
    logger.debug(f"Agent's maintenance margin: {maintenance_margin}")
    m_ratio = unhealthy_agent.mRatio(unhealthy_agent.position)
    logger.debug(f"Agent's mRatio: {m_ratio}")
    mark_to_market = unhealthy_agent.mark_to_market(unhealthy_agent.position, log=False)
    logger.debug(f"Agent's mark to market value: {mark_to_market}")
    market_tick = kernel.book.get_twap()
    logger.debug(f"Agent's market tick: {market_tick}")
    health_status = unhealthy_agent.is_healthy()
    assert not health_status, f"Expected health status to be False, got {health_status}"
    logger.debug(f"Agent's health status: {health_status}")
    
    # Log positions before liquidation
    logger.debug(f"Agent's position before liquidation: {unhealthy_agent.position}")
    logger.debug(f"Liquidator's position before liquidation: {liquidator.position}")
    
    # Add the unhealthy agent to the liquidator's watch list
    logger.debug("Adding unhealthy agent to Liquidator's watch list")
    liquidator.watch_list = [unhealthy_agent.id]
    
    # Perform liquidation
    current_time = 0
    logger.debug(f"Waking up liquidator at time: {current_time}")
    liquidator.wakeup(current_time)
    logger.debug("Liquidator wakeup complete")
    
    message = QuerySpreadResponseMsg(
        symbol=liquidator.symbol,
        bids=liquidator.known_bids[liquidator.symbol],
        asks=liquidator.known_asks[liquidator.symbol],
        mkt_closed=False,
        depth=2,
        last_trade=None,
    )
    logger.debug(f"Creating QuerySpreadResponseMsg: {message}")
    
    logger.debug(f"Liquidator receiving message at time: {current_time}")
    liquidator.receive_message(current_time, sender_id=0, message=message)
    logger.debug("Liquidator received message and updated known spreads")
    
    # Liquidator checks and performs liquidation
    logger.debug("Liquidator checking for agents to liquidate")
    liquidator.check_liquidate(unhealthy_agent)
    logger.debug("Liquidator performed liquidation check")
    
    # Log positions after liquidation
    logger.debug(f"Agent's position after liquidation: {unhealthy_agent.position}")
    logger.debug(f"Liquidator's position after liquidation: {liquidator.position}")
    
    # Verify agent's health after liquidation
    logger.debug("Re-evaluating agent's health after liquidation")
    new_m_ratio = unhealthy_agent.mRatio(unhealthy_agent.position)
    logger.debug(f"Agent's new mRatio: {new_m_ratio}")
    new_health_status = unhealthy_agent.is_healthy()
    logger.debug(f"Agent's new health status: {new_health_status}")
    
    # Assertions to verify the liquidation process
    logger.debug("Asserting the results of liquidation")
    assert unhealthy_agent.position["SIZE"] == 0, f"Expected SIZE to be 0, got {unhealthy_agent.position['SIZE']}"
    assert round(unhealthy_agent.position["COLLATERAL"],2) == 2.04, f"Expected COLLATERAL to be 2.040, got {unhealthy_agent.position['COLLATERAL']}"
    logger.debug("Test test_liquidator_calculate completed successfully")
