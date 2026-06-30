import numpy as np
from cryobednet.data.splits import region_holdout_split


def test_region_holdout_split():
    regions = np.array(["a", "a", "b", "b", "c", "c"])
    train, val, holdout = region_holdout_split(regions, "c", 0.25, 1)
    assert set(holdout) == {4, 5}
    assert not set(train).intersection(set(holdout))
    assert not set(val).intersection(set(holdout))
