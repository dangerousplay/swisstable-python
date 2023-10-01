import unittest
import random
from typing import Set

import pyximport

pyximport.install()

from hash_set import FlatHashSet


def generate_random_data(size: int) -> Set[int]:
    return set([random.randint(1, 1000000) for _ in range(size)])


class TestFlatHashSet(unittest.TestCase):
    def test_initialization(self):
        fhs = FlatHashSet()
        self.assertEqual(len(fhs._map._pairs), 16)

    def test_add(self):
        for _ in range(1000):
            fhs = FlatHashSet()
            data = generate_random_data(50)

            for _ in range(2):
                self.validate_add_all(data, fhs)

            self.assertEqual(len(fhs), len(data))

    def validate_add_all(self, data, fhs):
        for value in data:
            fhs.add(value)

            self.assertIn(value, fhs)

    def test_remove_all(self):
        for _ in range(1000):
            fhs = FlatHashSet()

            data = generate_random_data(50)

            for value in data:
                fhs.add(value)

            for i, value in enumerate(data):
                fhs.remove(value)
                self.assertNotIn(value, fhs)
                self.assertEqual(len(fhs), len(data) - 1 - i)

    def test_remove_some(self):
        for _ in range(1000):
            fhs = FlatHashSet()
            retain = 10
            data = generate_random_data(50)

            for value in data:
                fhs.add(value)

            while len(data) > retain:
                value = data.pop()
                fhs.remove(value)
                self.assertNotIn(value, fhs)

            for i, value in enumerate(data):
                self.assertIn(value, fhs)

            self.assertEqual(len(fhs), retain)


    def test_intersection(self):
        fhs1 = FlatHashSet()
        fhs2 = FlatHashSet()

        amount = 100
        amount_second = int(amount/2)

        for i in range(amount):
            fhs1.add(i)

        for i in range(amount_second):
            fhs2.add(i)

        fhs3 = fhs1.intersect(fhs2)

        for i in range(amount_second):
            self.assertIn(i, fhs3)

        self.assertEqual(len(fhs3), amount_second)

    def test_union(self):
        fhs1 = FlatHashSet()
        fhs2 = FlatHashSet()

        amount = 100
        expected_amount = amount * 2

        for i in range(1, amount+1):
            fhs1.add(i)

        for i in range(amount * 2, amount, -1):
            fhs2.add(i)

        fhs3 = fhs1.union(fhs2)

        for i in range(1, expected_amount):
            self.assertIn(i, fhs3)

        self.assertEqual(len(fhs3), expected_amount)


if __name__ == '__main__':
    unittest.main()
