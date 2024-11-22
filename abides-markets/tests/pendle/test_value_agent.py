import logging
import queue
import numpy as np
from abides_core.agent import Agent
from abides_core.rate_oracle import RateOracle
from abides_core import NanosecondTime
from abides_markets.agents.value_agent import ValueAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import Side
from datetime import datetime
from abides_core.utils import str_to_ns, merge_swap, fmt_ts
from abides_core import Message, NanosecondTime
from typing import Any, Dict, List, Optional, Tuple, Type

from abides_core.message import Message, MessageBatch, WakeupMsg, SwapMsg, UpdateRateMsg
logger = logging.getLogger(__name__)


class FakeOrderBook:
    def __init__(self):
        self.last_twap = -3.100998
    
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

def test_value_agent():

    agent_id = 1
    symbol = "PEN"
    collateral = 100_000
    wake_up_freq = 600_000_000  
    r_bar = 0.10 
    coef = [0.05, 0.40]  


    random_state = np.random.RandomState(seed=42)


    value_agent = ValueAgent(
        id=agent_id,
        symbol=symbol,
        random_state=random_state,
        collateral=collateral,
        wake_up_freq=wake_up_freq,
        r_bar=r_bar,
        coef=coef
    )

    kernel = FakeKernel()
    class MockRateOracle:
        def get_floating_rate(self, current_time):
            return 0.02  # 返回固定的融资利率

    mock_rate_oracle = MockRateOracle()
    value_agent.rate_oracle = mock_rate_oracle
    value_agent.kernel = kernel

    value_agent.mkt_open = 1
    value_agent.mkt_close = 1_000_000_000  
    value_agent.current_time = 2

    value_agent.exchange_id = 0

    value_agent.known_bids = {symbol: [(1000, 50)]}
    value_agent.known_asks = {symbol: [(1010, 50)]}


    placed_orders = []
    logger.debug("Initialized list to capture placed orders")

    def mock_place_limit_order(symbol, quantity, side, price):
        logger.debug(f"Placed limit order - Symbol: {symbol}, Quantity: {quantity}, Side: {side}, Price: {price}")
        placed_orders.append({'symbol': symbol, 'quantity': quantity, 'side': side, 'price': price})

    value_agent.place_limit_order = mock_place_limit_order
    logger.debug("Replaced agent's place_limit_order method with mock method")


    def mock_observe_price(symbol, current_time, random_state):
        return 1005 

    def mock_get_floating_rate(current_time):
        return 0.02 

    kernel.driving_oracle = type('MockOracle', (), {'observe_price': mock_observe_price})
    kernel.rate_oracle = type('MockOracle', (), {'get_floating_rate': mock_get_floating_rate})

    logger.debug("Simulating agent's wakeup call")
    value_agent.wakeup(value_agent.current_time)

    logger.debug("Simulating receipt of QuerySpreadResponseMsg")
    message = QuerySpreadResponseMsg(
        symbol=symbol,
        bids=value_agent.known_bids[symbol],
        asks=value_agent.known_asks[symbol],
        mkt_closed=False,
        depth=1,
        last_trade=None,
    )
    
    class MockDrivingOracle:
        def observe_price(self, symbol, current_time, random_state):
            return 1005  


    mock_driving_oracle = MockDrivingOracle()


    value_agent.driving_oracle = mock_driving_oracle

    logger.debug("Agent receiving the QuerySpreadResponseMsg")
    value_agent.receive_message(value_agent.current_time, sender_id=0, message=message)

    assert len(placed_orders) == 1, f"Expected 1 order to be placed, but got {len(placed_orders)}"

    order = placed_orders[0]
    logger.debug(f"Order placed: {order}")


    assert order['side'] == Side.ASK, f"Expected order side to be ASK, but got {order['side']}"
    assert order['price'] == 1000, f"Expected order price to be 1000, but got {order['price']}"

    logger.info("ValueAgent test passed successfully.")

