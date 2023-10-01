import math
from typing import List, TypeVar, TypeAlias, Tuple, Optional
from enum import Enum

import sse_match


GROUP_SIZE = 16
MAX_AVG_GROUP_LOAD = 14


class Mask(Enum):
    HASH_1 = 0xFFFF_FFFF_FFFF_FF80
    HASH_2 = 0x0000_0000_0000_007F
    EMPTY = 0b1000_0000             # -128
    TOMBSTONE = 0b1111_1110         # -2


def num_groups(size: int) -> int:
    groups = int(math.floor(size / MAX_AVG_GROUP_LOAD))

    if groups == 0:
        groups = 1

    return groups


def split_hash(h: int) -> Tuple[int, int]:
    return (h & Mask.HASH_1.value) >> 7, h & Mask.HASH_2.value


def group_index(n: int) -> int:
    return int(math.floor(n / GROUP_SIZE)) * GROUP_SIZE


K = TypeVar('K')
V = TypeVar('V')


class FlatHashMap:
    _control: List[Optional[int]]
    _pairs: List[Optional[Tuple[K, V]]]
    _resident_keys: int
    _removed_keys: int
    _limit: int

    def __init__(self, capacity=16):
        self._initialize_control_(capacity)
        self._initialize_pairs_(capacity)
        self._resident_keys = 0
        self._removed_keys = 0
        self._limit = num_groups(capacity) * MAX_AVG_GROUP_LOAD

    def _initialize_control_(self, capacity: int):
        self._control = [Mask.EMPTY.value] * capacity

    def _initialize_pairs_(self, capacity: int):
        self._pairs = [None] * capacity

    def _find_(self, key: K) -> Tuple[bool, Optional[V], int]:
        hi, lo = split_hash(key.__hash__())
        probe_index = hi % self._pairs.__len__()

        while True:
            keys = self._control_keys_at_index_(probe_index)

            matches = sse_match.find_matches(lo, keys)

            while matches != 0:
                match, bitmask = sse_match.next_match(matches)
                index = probe_index + match

                if key == self._pairs[index][0]:
                    return True, self._pairs[index][1], index

                matches = bitmask

            matches = sse_match.find_matches(Mask.EMPTY.value, keys)

            if matches != 0:
                match, _ = sse_match.next_match(matches)
                index = probe_index + match

                return False, None, index

            probe_index += GROUP_SIZE

            if probe_index >= self._pairs.__len__():
                probe_index = 0

    def __contains__(self, key):
        found, _, _ = self._find_(key)

        return found

    def _next_size_(self):
        n = len(self._pairs) * 2

        if self._removed_keys >= self._resident_keys:
            n = len(self._pairs)

        return int(n)

    def _rehash_(self, n: int):
        pairs = self._pairs
        control = self._control

        self._initialize_pairs_(n)
        self._initialize_control_(n)
        self._limit = num_groups(n) * MAX_AVG_GROUP_LOAD
        self._removed_keys = 0
        self._resident_keys = 0

        for k, v in filter(lambda x: x is not None, pairs):
            self.put(k, v)

        del pairs
        del control

    def __getitem__(self, item: K):
        _, value, _ = self._find_(item)

        return value

    def put(self, key: K, value: V) -> Optional[V]:
        if self._resident_keys >= self._limit:
            self._rehash_(self._next_size_())

        hi, lo = split_hash(key.__hash__())
        found, old_value, index = self._find_(key)

        if found:
            self._pairs[index] = (key, value)
        else:
            self._pairs[index] = (key, value)
            self._control[index] = lo
            self._resident_keys += 1

        return old_value

    def remove(self, key: K) -> Optional[V]:
        found, value, index = self._find_(key)

        if not found:
            return None

        keys = self._control_keys_at_index_(group_index(index))

        matches = sse_match.find_matches(Mask.EMPTY.value, keys)

        if matches != 0:
            self._control[index] = Mask.EMPTY.value
        else:
            self._control[index] = Mask.TOMBSTONE.value
            self._removed_keys += 1

        self._resident_keys -= 1
        self._pairs[index] = None

        return value

    def __delitem__(self, key: K):
        self.remove(key)

    def __setitem__(self, key: K, value: V):
        self.put(key, value)

    def _control_keys_at_index_(self, probe_index):
        keys = self._control[probe_index:]

        if len(keys) < GROUP_SIZE:
            keys += [Mask.TOMBSTONE.value] * (GROUP_SIZE - len(keys))

        return bytes(keys)

    def __len__(self):
        return self._resident_keys
