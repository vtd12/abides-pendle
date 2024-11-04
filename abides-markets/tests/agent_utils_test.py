import numpy as np

from abides_core.generators import PoissonTimeGenerator
from abides_markets.agents.utils import tick_to_rate, rate_to_tick

def test_poisson_time_generator():
    gen = PoissonTimeGenerator(
        lambda_time=2, random_generator=np.random.RandomState(seed=1)
    )

    for _ in range(10):
        print(gen.next())

def test_convertion():
    assert tick_to_rate(0)==0
    assert round(tick_to_rate(1), 4)==0.0001
    assert round(tick_to_rate(-1), 4)==-0.0001
    assert round(tick_to_rate(2), 4)==0.0002
    assert round(tick_to_rate(-2), 4)==-0.0002

    assert rate_to_tick(0)==0
    assert rate_to_tick(0.0001) == 1
    assert rate_to_tick(-0.0001) == -1
    assert rate_to_tick(0.0002) == 2
    assert rate_to_tick(-0.0002) == -2