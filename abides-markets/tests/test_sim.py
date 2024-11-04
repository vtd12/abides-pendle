import shutil
import pandas as pd
import numpy as np

from abides_core import Kernel
from abides_core.utils import subdict, parse_logs_df
# from abides_markets.configs.rmsc04 import build_config as build_config_rmsc04
from abides_markets.configs.prmsc1 import build_config as build_config_prmsc1


def test_prmsc1():
    config = build_config_prmsc1(
        seed=1,
        book_logging=False,
        log_orders=True,
        exchange_log_orders=False,
    )

    kernel_seed = np.random.randint(low=0, high=2 ** 32, dtype="uint64")

    kernel = Kernel(
        log_dir="__test_logs",
        random_state=np.random.RandomState(seed=kernel_seed),
        **subdict(
            config,
            [
                "start_time",
                "stop_time",
                "swap_interval",
                "agents",
                "agent_latency_model",
                "default_computation_delay",
                "custom_properties"
            ],
        ),
        skip_log=True,
    )

    end_state = kernel.run()

    df = parse_logs_df(end_state)
    # Convert the DataFrame to Markdown format
    df_markdown = df.to_markdown(index=False)

    # Save the Markdown string to a text file
    with open('log/test_log.txt', 'w') as f:
        f.write(df_markdown)

    shutil.rmtree("log/__test_logs")
    ## just checking simulation runs without crashing and reaches the assert
    assert True