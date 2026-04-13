import numpy as np

from training import _iter_resnet_epoch_batches, _resnet_steps_per_epoch


def test_resnet_epoch_batches_include_remainder_batch():
    x = np.arange(10)
    batch_size = 4
    steps_ep = _resnet_steps_per_epoch(len(x), batch_size)

    batches = [
        {"image": x[0:4], "label": x[0:4]},
        {"image": x[4:8], "label": x[4:8]},
        {"image": x[8:10], "label": x[8:10]},
    ]

    yielded = list(_iter_resnet_epoch_batches(iter(batches), steps_ep))

    assert steps_ep == 3
    assert len(yielded) == 3
    assert yielded[-1]["image"].shape[0] == 2


def test_resnet_epoch_steps_match_exact_divisibility():
    assert _resnet_steps_per_epoch(12, 4) == 3
