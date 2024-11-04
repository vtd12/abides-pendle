import datetime as dt
import logging
from math import sqrt
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pickle

import os
import requests

from abides_core import NanosecondTime
from abides_core.rate_oracle import RateOracle
from abides_markets.agents.utils import tick_to_rate


logger = logging.getLogger(__name__)


class BTCOracle(RateOracle):
    """
    This oracle return a constant floating rate overtime.
    """

    def __init__(
        self,
        mkt_open: NanosecondTime = 0,
        mkt_close: NanosecondTime = 0,
    ) -> None:
        # Base class init.
        super().__init__(mkt_open, mkt_close)

        self.current_time = mkt_open

        if not os.path.exists("BTC-Binance-2023.pkl"):
            self.funding_rate_table = self.download_fundingrate_data()
        else: 
            with open("BTC-Binance-2023.pkl", 'rb') as f:
                self.funding_rate_table = pickle.load(f)
        
        for i in range(len(self.funding_rate_table)):
            self.funding_rate_table.iloc[i, 1] = int(self.funding_rate_table.iloc[i, 1].timestamp()*1000)

        self.last_rate: float = 0

    def get_floating_rate(self, current_time: NanosecondTime) -> float:
        """
        Get the floating rate of the time from the table. This rate is not normalized yet.

        Arguments:
            current_time: The time that this oracle is queried for the floating rate

        Returns:
            The corresponding floating rate
        """
        super().get_floating_rate(current_time)
        query_index = 0

        while  self.funding_rate_table.iloc[query_index, 1] < self.current_time:
            self.last_rate = self.funding_rate_table.iloc[query_index, 0]
            query_index += 1
            
        self.funding_rate_table.drop(self.funding_rate_table.index[:query_index], inplace=True)

        return self.last_rate
    
    def download_fundingrate_data(self):
        url = "https://raw.githubusercontent.com/vtd12/funding-rate/main/BTC-Binance-2023.pkl"
        
        response = requests.get(url)

        if response.status_code == 200:
            with open("BTC-Binance-2023.pkl", "wb") as f:
                f.write(response.content)
            with open("BTC-Binance-2023.pkl", 'rb') as f:
                data = pickle.load(f)
                return data
        else:
            logger.error("Request failed with status code:", response.status_code)


    