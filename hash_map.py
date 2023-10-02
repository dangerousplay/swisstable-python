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
    TOMBSTONE = 0b1111_1110         # 254


def num_groups(size: int) -> int:
    """
    :param size: An integer representing the total size
    :return: An integer representing the number of groups
    """
    groups = int(math.floor(size / MAX_AVG_GROUP_LOAD))

    if groups == 0:
        groups = 1

    return groups


def split_hash(h: int) -> Tuple[int, int]:
    return (h & Mask.HASH_1.value) >> 7, h & Mask.HASH_2.value


K = TypeVar('K')
V = TypeVar('V')


def _ensure_multiple_(multiple_of: int, number: int):
    """
    :param number: The new capacity.
    :return: The adjusted number that ensures it is a multiple of multiple_of.
    """
    cap_mod_g = number % multiple_of

    if cap_mod_g != 0:
        number += multiple_of - cap_mod_g

    return number


class FlatHashMap:
    _control: List[Optional[int]]
    _pairs: List[Optional[Tuple[K, V]]]
    _resident_keys: int
    _removed_keys: int
    _limit: int

    def __init__(self, capacity=16):
        capacity = _ensure_multiple_(GROUP_SIZE, capacity)

        self._initialize_control_(capacity)
        self._initialize_pairs_(capacity)
        self._resident_keys = 0
        self._removed_keys = 0
        self._limit = num_groups(capacity) * MAX_AVG_GROUP_LOAD

    def _initialize_control_(self, capacity: int):
        self._control = [Mask.EMPTY.value] * capacity

    def _initialize_pairs_(self, capacity: int):
        self._pairs = [None] * capacity

    def _find_(self, key: K) -> Tuple[bool, Optional[V], int, int]:
        """
        Find the given key.

        :param key: The key to search for in the FlatHashMap.
        :return: A tuple containing four elements
            - a boolean indicating if the key was found,
            - an optional value associated with the key (if found)
            - the index of the key-value pair
            - the probe index of the match
        """

        hi, lo = split_hash(key.__hash__())
        probe_index = self._key_group_(hi) * GROUP_SIZE

        while True:
            keys = self._control_keys_at_index_(probe_index)

            matches = sse_match.find_matches(lo, keys)

            while matches != 0:
                match, bitmask = sse_match.next_match(matches)
                index = (probe_index + match) % self._pairs.__len__()

                if key == self._pairs[index][0]:
                    return True, self._pairs[index][1], index, probe_index

                matches = bitmask

            matches = sse_match.find_matches(Mask.EMPTY.value, keys)

            if matches != 0:
                match, _ = sse_match.next_match(matches)
                index = (probe_index + match) % self._pairs.__len__()

                return False, None, index, probe_index

            probe_index += GROUP_SIZE

            if probe_index >= self._pairs.__len__():
                probe_index = 0

    def __contains__(self, key):
        found, _, _, _ = self._find_(key)

        return found

    def _key_group_(self, key: int):
        return key % int(self._pairs.__len__() / GROUP_SIZE)

    def _next_size_(self):
        n = len(self._pairs) * 2

        if self._removed_keys >= self._resident_keys:
            n = len(self._pairs)

        return _ensure_multiple_(GROUP_SIZE, int(n))

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
        _, value, _, _ = self._find_(item)

        return value

    def put(self, key: K, value: V) -> Optional[V]:
        """
        :param key: The key to be inserted into the hash map.
        :param value: The value corresponding to the key to be inserted into the hash map.
        :return: The old value if the key already exists in the hash map, otherwise None.

        This method inserts a key-value pair into the FlatHashMap. If the number of resident keys in the FlatHashMap
        exceeds the limit, the hash map is rehashed to accommodate more keys. The method splits the hash of the key
        into high and low bytes, searches for the key in the hash map, and updates the value if the key already
        exists. If the key is not found, the key-value pair is inserted into the hash map along with the control byte
        and the resident keys count is incremented. Finally, the method returns the old value if the key already
        exists in the hash map, otherwise it returns None.
        """
        if self._resident_keys >= self._limit:
            self._rehash_(self._next_size_())

        hi, lo = split_hash(key.__hash__())
        found, old_value, index, _ = self._find_(key)

        if found:
            self._pairs[index] = (key, value)
        else:
            self._pairs[index] = (key, value)
            self._control[index] = lo
            self._resident_keys += 1

        return old_value

    def remove(self, key: K) -> Optional[V]:
        """
        :param key: The key to be removed from the FlatHashMap.
        :return: The value associated with the specified key, or None if the key is not found.
        """
        if self._removed_keys > len(self._pairs) / 2:
            self._rehash_(len(self._pairs))

        found, value, index, probe_index = self._find_(key)

        if not found:
            return None

        keys = self._control_keys_at_index_(probe_index)

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

    def __iter__(self):
        return filter(lambda x: x is not None, self._pairs)

    def __add__(self, other):
        for k, v in other:
            self.put(k, v)

        return self

    def __sub__(self, other):
        for k, v in other:
            self.remove(k)

        return self

    def __str__(self):
        return f"{list(self.__iter__())}, cap={self._pairs.__len__()}"

    def __repr__(self):
        return self.__str__()

    def _control_keys_at_index_(self, probe_index):
        keys = self._control[probe_index:]

        if len(keys) < GROUP_SIZE:
            amount_missing = GROUP_SIZE - len(keys)
            keys += self._control[0:amount_missing]

        return bytes(keys)

    def __len__(self):
        return self._resident_keys
