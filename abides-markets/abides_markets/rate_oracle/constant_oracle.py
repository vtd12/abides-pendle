import datetime as dt
import logging
from math import sqrt
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from abides_core import NanosecondTime
from abides_core.rate_oracle import RateOracle
from ..agents.utils import tick_to_rate


logger = logging.getLogger(__name__)


class ConstantOracle(RateOracle):
    """
    This oracle return a constant floating rate overtime.
    """

    def __init__(
        self,
        mkt_open: NanosecondTime = 0,
        mkt_close: NanosecondTime = 0,
        const_rate: float = 0,
        starting_market_rate = 0
    ) -> None:
        # Base class init.
        super().__init__(mkt_open, starting_market_rate)

        self.const_rate = const_rate
        

    def get_floating_rate(self, current_time: NanosecondTime) -> float:
        """
        Arguments:
            current_time: The time that this oracle is queried for the floating rate

        Returns:
            Constant but random floating rate
        """
        super().get_floating_rate(current_time)

        np.random.seed(current_time%2**32)

        return np.random.uniform(0, 0.0002)
    
    def get_daily_open_price(
        self, symbol: str, mkt_open: NanosecondTime
    ) -> int:
        return self.const_rate
    