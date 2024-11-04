import datetime as dt
import logging
from math import sqrt
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from abides_core import NanosecondTime


logger = logging.getLogger(__name__)

from .utils import fmt_ts, str_to_ns


class RateOracle:
    """
    The Base Rate Oracle: output the floating rate for swaps periodically, provided by a third party

    Attributes:
        id: Must be a unique number (usually autoincremented).
        name: For human consumption, should be unique (often type + number).
        log_events: flag to log or not the events during the simulation
        log_to_file: flag to write on disk or not the logged events
    """

    def __init__(
        self,
        start_time: NanosecondTime,
        starting_market_rate: float,
    ) -> None:
        self.current_time: NanosecondTime = start_time
        self.market_rate: float = starting_market_rate

    def get_floating_rate(
            self,
            current_time
    ) -> None:
        """
        Called by trading agents periodically to swap with their fix rate

        Arguments:
            current_time: The time when the floating rate is queried.
        """
        assert current_time >= self.current_time
        
        self.current_time = current_time

        logger.debug(f"Query Pendle Oracle at {current_time}")

    def update_market_rate(
        self, new_mid: float, update_time: NanosecondTime
    ) -> None:
        pass