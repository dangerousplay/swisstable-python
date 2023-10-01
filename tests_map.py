import math
import random
import unittest
from typing import Set

import pyximport
pyximport.install()

from hash_map import FlatHashMap

import sse_match


class ColisionHash:
    def __init__(self, value, hash=int(math.pow(10, 16))):
        self.value = value
        self.hash = hash

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return f"({self.hash}:{self.value})"


def rand_key(i: int):
    return f"{i}-{random.randint(10, 100)}"


def generate_random_data(amount: int) -> Set[int]:
    return set([random.randint(1, 10000000) for _ in range(amount)])


class SSETests(unittest.TestCase):
    def test_trailing_zeros(self):
        for i in range(15):
            expected = i+1
            number = math.pow(10, expected)

            actual = sse_match.trailing_zeros(number)

            self.assertEqual(actual, expected)

        self.assertEqual(sse_match.trailing_zeros(0), 16)

    def test_next_match(self):
        expected_matches = [2, 3, 5, 7, 11]

        bitmask = 0

        for match in expected_matches:
            bitmask += math.pow(2, match)

        for match in expected_matches:
            actual, next_mask = sse_match.next_match(bitmask)

            self.assertEqual(actual, match)
            self.assertEqual(next_mask, bitmask - math.pow(2, match))

            bitmask = next_mask

        actual, next_mask = sse_match.next_match(bitmask)

        self.assertEqual(actual, 16)
        self.assertEqual(next_mask, 0)


class FlatMapTests(unittest.TestCase):
    def test_put_no_collision(self):
        hash_map = FlatHashMap()

        key = 'a'
        value = 1

        hash_map[key] = value

        self.assertEqual(hash_map.__len__(), 1)
        self.assertIn(key, hash_map)

    def test_put_new_value(self):
        hash_map = FlatHashMap()

        key = 'a'
        value = 1
        new_value = 2

        hash_map[key] = value
        hash_map[key] = new_value

        self.assertEqual(hash_map.__len__(), 1)
        self.assertIn(key, hash_map)

    def test_put_collision(self):
        hash_map = FlatHashMap()

        amount = 10
        previous_keys = []

        for i in range(amount):
            key = ColisionHash(i)
            hash_map[key] = i

            self.assertEqual(hash_map.__len__(), i + 1)

            previous_keys.append(key)

            for key in previous_keys:
                self.assertIn(key, hash_map)

        self.assertEqual(hash_map.__len__(), amount)

    def test_put_rehash(self):
        for _ in range(50):
            hash_map = FlatHashMap()

            amount = 250
            previous_keys = []

            for i, data in enumerate(generate_random_data(amount)):
                hash_map[data] = data

                self.assertEqual(hash_map.__len__(), i + 1)

                previous_keys.append(data)

                for key in previous_keys:
                    self.assertIn(key, hash_map)

            self.assertEqual(hash_map.__len__(), amount)

    def test_delete_collision(self):
        hash_map = FlatHashMap()

        amount = 10
        previous_keys = []

        for i in range(amount):
            key = ColisionHash(i)
            hash_map[key] = i

            previous_keys.append((key, key.value))

        self.validate_remove(hash_map, previous_keys)

    def test_delete_all(self):
        for _ in range(100):
            hash_map = FlatHashMap(capacity=32)

            amount = 32
            previous_keys = []

            for i in generate_random_data(amount):
                key = rand_key(i)
                hash_map[key] = i

                previous_keys.append((key, i))

            self.validate_remove(hash_map, previous_keys)

    def test_delete_some(self):
        for _ in range(100):
            hash_map = FlatHashMap(capacity=32)

            amount = 32
            previous_keys = []

            for i in generate_random_data(amount):
                key = rand_key(i)
                hash_map[key] = i

                previous_keys.append((key, i))

            self.validate_remove(hash_map, previous_keys, retain=10)

    def validate_remove(self, hash_map, previous_keys, retain=0):
        initial_size = hash_map.__len__()
        removed = 1

        while len(previous_keys) > retain:
            key, value = previous_keys.pop()

            actual_value = hash_map.remove(key)

            self.assertEqual(actual_value, value)
            self.assertNotIn(key, hash_map)
            self.assertEqual(hash_map.__len__(), initial_size - removed)

            for previous_key, _ in previous_keys:
                self.assertIn(previous_key, hash_map)

            removed += 1

        for key, value in previous_keys:
            self.assertIn(key, hash_map)

            actual_value = hash_map[key]
            self.assertEqual(actual_value, value)

        self.assertEqual(hash_map.__len__(), retain)


if __name__ == '__main__':
    unittest.main()
