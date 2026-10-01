import numpy as np

from fair_calibrate.chunks import map_chunks

DATA = np.arange(1000, dtype=float) ** 2


def square_root_chunk(start, stop):
    return np.sqrt(DATA[start:stop])


def test_single_worker_is_one_call():
    calls = []

    def record(start, stop):
        calls.append((start, stop))
        return np.zeros(stop - start)

    map_chunks(record, 10, workers=1)
    assert calls == [(0, 10)]


def test_parallel_matches_serial_in_order():
    serial = np.concatenate(map_chunks(square_root_chunk, len(DATA), workers=1))
    parallel = np.concatenate(map_chunks(square_root_chunk, len(DATA), workers=3))
    assert np.array_equal(serial, parallel)


def test_uneven_and_tiny_sizes():
    for n in (1, 2, 7, 13):
        out = np.concatenate(map_chunks(square_root_chunk, n, workers=4))
        assert np.array_equal(out, np.sqrt(DATA[:n]))


def test_empty():
    assert map_chunks(square_root_chunk, 0, workers=4) == []
