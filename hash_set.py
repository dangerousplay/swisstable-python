from typing import TypeVar

from hash_map import FlatHashMap


V = TypeVar('V')


class FlatHashSet:
    def __init__(self, capacity=16):
        self._map = FlatHashMap(capacity=capacity)

    def add(self, value: V):
        self._map.put(value, None)

    def __contains__(self, item):
        return self._map.__contains__(item)

    def remove(self, value: V):
        self._map.remove(value)

    def __add__(self, other):
        return self.union(other)

    def __sub__(self, other):
        hash_set = FlatHashSet()
        hash_set._map += self._map
        hash_set._map -= other._map

        return hash_set

    def __len__(self):
        return len(self._map)

    def intersection(self, other):
        hash_set = FlatHashSet()

        for k, v in self._map:
            if k in other:
                hash_set.add(k)

        return hash_set

    def union(self, other):
        hash_set = FlatHashSet()
        hash_set._map += other._map
        hash_set._map += self._map

        return hash_set

    def __str__(self):
        keys = [k.__str__() for k, _ in self._map]

        return ", ".join(keys)

    def __repr__(self):
        return self.__str__()
