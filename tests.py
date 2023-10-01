import math
import random
import unittest
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
        self.assertTrue(hash_map.__contains__(key))

    def test_put_new_value(self):
        hash_map = FlatHashMap()

        key = 'a'
        value = 1
        new_value = 2

        hash_map[key] = value
        hash_map[key] = new_value

        self.assertEqual(hash_map.__len__(), 1)
        self.assertTrue(hash_map.__contains__(key))

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
                self.assertTrue(hash_map.__contains__(key))

        self.assertEqual(hash_map.__len__(), amount)

    def test_put_rehash(self):
        hash_map = FlatHashMap()

        amount = 250
        previous_keys = []

        for i in range(amount):
            key = rand_key(i)
            hash_map[key] = i

            self.assertEqual(hash_map.__len__(), i + 1)

            previous_keys.append(key)

            for key in previous_keys:
                self.assertTrue(hash_map.__contains__(key))

        self.assertEqual(hash_map.__len__(), amount)

    def test_delete_collision(self):
        hash_map = FlatHashMap()

        amount = 10
        previous_keys = []

        for i in range(amount):
            key = ColisionHash(i)
            hash_map[key] = i

            previous_keys.append((key, key.value))

        self.validate_remove_all(hash_map, previous_keys)

    def test_delete(self):
        hash_map = FlatHashMap(capacity=32)

        amount = 10
        previous_keys = []

        for i in range(amount):
            key = rand_key(i)
            hash_map[key] = i

            previous_keys.append((key, i))

        self.validate_remove_all(hash_map, previous_keys)

    def validate_remove_all(self, hash_map, previous_keys):
        initial_size = hash_map.__len__()
        removed = 1

        while len(previous_keys) > 0:
            key, value = previous_keys.pop()

            actual_value = hash_map.remove(key)

            self.assertEqual(actual_value, value)
            self.assertFalse(hash_map.__contains__(key))
            self.assertEqual(hash_map.__len__(), initial_size - removed)

            for previous_key, _ in previous_keys:
                self.assertTrue(hash_map.__contains__(previous_key))

            removed += 1

        self.assertEqual(hash_map.__len__(), 0)


if __name__ == '__main__':
    unittest.main()
