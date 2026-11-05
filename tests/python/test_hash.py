import pytest
import numpy as np
import concurrent.futures
import algoat
from algoat import hash as ahash


def test_xxhash3_64_test_vectors():
    # Empty string
    assert ahash.xxhash3_64("") == 0x2D06800538D394C2
    assert ahash.xxhash3_64(b"") == 0x2D06800538D394C2
    assert ahash.xxhash3_64("", seed=0x9E3779B185EBCA8D) == 0xA8A6B918B2F0364A

    # String vs bytes vs bytearray vs memoryview vs numpy array
    text = "hello world hashing with algoat xxhash3"
    b_text = text.encode("utf-8")
    arr = np.frombuffer(b_text, dtype=np.uint8)

    h_str = ahash.xxhash3_64(text)
    h_bytes = ahash.xxhash3_64(b_text)
    h_bytearray = ahash.xxhash3_64(bytearray(b_text))
    h_memview = ahash.xxhash3_64(memoryview(b_text))
    h_arr = ahash.xxhash3_64(arr)

    assert h_str == h_bytes
    assert h_str == h_bytearray
    assert h_str == h_memview
    assert h_str == h_arr


def test_xxhash3_128_test_vectors():
    # Empty string
    h128_0 = ahash.xxhash3_128("")
    assert isinstance(h128_0, tuple)
    assert len(h128_0) == 2
    assert h128_0 == (0x6001C324468D497F, 0x99AA06D3014798D8)

    # Seeded
    h128_s = ahash.xxhash3_128("", seed=0x9E3779B185EBCA8D)
    assert h128_s == (0xA986DFC5D7605BFE, 0x00FEAA732A3CE25E)


def test_wyhash_test_vectors():
    assert ahash.wyhash("") == 0x93228A4DE0EEC5A2
    assert ahash.wyhash("a") == 0xACED12527FE5BFF8
    assert ahash.wyhash("abc") == 0x989B4A209C1011C9
    assert ahash.wyhash("message digest") == 0x309AB4C045215E8F

    # Seeded
    assert ahash.wyhash("a", seed=1) == 0xC5BAC3DB178713C4
    assert ahash.wyhash("abc", seed=2) == 0xA97F2F7B1D9B3314


def test_highway_hash_test_vectors():
    key = (1, 2, 3, 4)
    # Empty string
    assert ahash.highway_hash_64("", key) == 0x92943A34E3447FAA
    h128_empty = ahash.highway_hash_128("", key)
    assert isinstance(h128_empty, tuple)
    assert len(h128_empty) == 2

    h256_empty = ahash.highway_hash_256("", key)
    assert len(h256_empty) == 4
    assert h256_empty == (
        0x5ABC20E8DB09EBD6,
        0x58DECE8699205080,
        0x9DE9645881E6786E,
        0x1B00FEFCC21128C2,
    )

    # 1 byte 0xFF
    assert ahash.highway_hash_64(b"\xff", key) == 0x7858F24D2D79B2B2
    h128_1b = ahash.highway_hash_128(b"\xff", key)
    assert len(h128_1b) == 2

    assert ahash.highway_hash_256(b"\xff", key) == (
        0x7161CADBF7CD70E1,
        0xAAC4905DE62B2F5E,
        0x07B02B936933FAA7,
        0xC8EFCFC45B239F8D,
    )

    # 32 sequential bytes
    seq32 = bytes(range(32))
    assert ahash.highway_hash_64(seq32, key) == 0x6CB9D962DB4F34E8
    h128_32b = ahash.highway_hash_128(seq32, key)
    assert len(h128_32b) == 2

    assert ahash.highway_hash_256(seq32, key) == (
        0xE9FA92165762737B,
        0xA637FF0FDEFC1F58,
        0x12A2DFAD57FB47B0,
        0xF8116C2CFC585FF6,
    )


def test_highway_key_validation():
    # Invalid key length
    with pytest.raises(ValueError, match="HighwayKey must contain exactly 4"):
        ahash.highway_hash_64("test", (1, 2, 3))

    # Invalid key type
    with pytest.raises(TypeError, match="HighwayKey must be a sequence"):
        ahash.highway_hash_64("test", 12345)


def test_invalid_input_type():
    with pytest.raises(TypeError, match="Expected a bytes-like object or str"):
        ahash.xxhash3_64(12345)

    with pytest.raises(TypeError, match="Expected a bytes-like object or str"):
        ahash.wyhash({})


def test_multithreading_gil_release():
    payload = b"benchmark payload " * 1000
    key = [1, 2, 3, 4]

    def work(seed):
        h1 = ahash.xxhash3_64(payload, seed=seed)
        h2 = ahash.wyhash(payload, seed=seed)
        h3 = ahash.highway_hash_64(payload, key)
        return h1, h2, h3

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(work, s) for s in range(20)]
        results = [f.result() for f in futures]

    assert len(results) == 20
    # Different seeds should give different hashes
    assert results[0][0] != results[1][0]
