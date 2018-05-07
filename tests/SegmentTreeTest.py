import unittest
import sys
sys.path.insert(0, "../src")
import numpy as np
from SegmentTree import SumSegmentTree, MinSegmentTree

class SegmentTreeTest(unittest.TestCase):

    def testSumSegmentTreeSum(self):
        tree = SumSegmentTree(4)
        tree[2] = 1.0
        tree[3] = 3.0

        self.assertTrue(np.isclose(tree.sum(), 4.0))
        self.assertTrue(np.isclose(tree.sum(0, 2), 0.0))
        self.assertTrue(np.isclose(tree.sum(0, 3), 1.0))
        self.assertTrue(np.isclose(tree.sum(2, 3), 1.0))
        self.assertTrue(np.isclose(tree.sum(2, -1), 1.0))
        self.assertTrue(np.isclose(tree.sum(2, 4), 4.0))

    def testSumSegmentTreeRepetitiveSetOverride(self):
        tree = SumSegmentTree(4)

        tree[2] = 1.0
        tree[2] = 3.0

        self.assertTrue(np.isclose(tree.sum(), 3.0))
        self.assertTrue(np.isclose(tree.sum(2, 3), 3.0))
        self.assertTrue(np.isclose(tree.sum(2, -1), 3.0))
        self.assertTrue(np.isclose(tree.sum(2, 4), 3.0))
        self.assertTrue(np.isclose(tree.sum(1, 2), 0.0))

        tree[2] = 0.0
        self.assertTrue(np.isclose(tree.sum(), 0.0))
        self.assertTrue(np.isclose(tree.sum(2, 3), 0.0))
        self.assertTrue(np.isclose(tree.sum(2, -1), 0.0))
        self.assertTrue(np.isclose(tree.sum(2, 4), 0.0))

    def testSumSegmentTreePrefixsumIdx1(self):
        tree = SumSegmentTree(4)

        tree[2] = 1.0
        tree[3] = 3.0
        self.assertTrue(tree.find_prefixsum_idx(0.0) == 2)
        self.assertTrue(tree.find_prefixsum_idx(0.5) == 2)
        self.assertTrue(tree.find_prefixsum_idx(0.99) == 2)
        self.assertTrue(tree.find_prefixsum_idx(1.01) == 3)
        self.assertTrue(tree.find_prefixsum_idx(3.00) == 3)
        self.assertTrue(tree.find_prefixsum_idx(4.00) == 3)

    def testSumSegmentTreePrefixsumIdx2(self):
        tree = SumSegmentTree(4)

        tree[0] = 0.5
        tree[1] = 1.0
        tree[2] = 1.0
        tree[3] = 3.0

        self.assertTrue(tree.find_prefixsum_idx(0.00) == 0)
        self.assertTrue(tree.find_prefixsum_idx(0.55) == 1)
        self.assertTrue(tree.find_prefixsum_idx(0.99) == 1)
        self.assertTrue(tree.find_prefixsum_idx(1.51) == 2)
        self.assertTrue(tree.find_prefixsum_idx(3.00) == 3)
        self.assertTrue(tree.find_prefixsum_idx(5.50) == 3)

    def testMinSegmentTreeIntervalTree(self):
        tree = MinSegmentTree(4)

        tree[0] = 1.0
        tree[2] = 0.5
        tree[3] = 3.0

        self.assertTrue(np.isclose(tree.min(), 0.5))
        self.assertTrue(np.isclose(tree.min(0, 2), 1.0))
        self.assertTrue(np.isclose(tree.min(0, 3), 0.5))
        self.assertTrue(np.isclose(tree.min(0, -1), 0.5))
        self.assertTrue(np.isclose(tree.min(2, 4), 0.5))
        self.assertTrue(np.isclose(tree.min(3, 4), 3.0))

        tree[2] = 0.7

        self.assertTrue(np.isclose(tree.min(), 0.7))
        self.assertTrue(np.isclose(tree.min(0, 2), 1.0))
        self.assertTrue(np.isclose(tree.min(0, 3), 0.7))
        self.assertTrue(np.isclose(tree.min(0, -1), 0.7))
        self.assertTrue(np.isclose(tree.min(2, 4), 0.7))
        self.assertTrue(np.isclose(tree.min(3, 4), 3.0))

        tree[2] = 4.0

        self.assertTrue(np.isclose(tree.min(), 1.0))
        self.assertTrue(np.isclose(tree.min(0, 2), 1.0))
        self.assertTrue(np.isclose(tree.min(0, 3), 1.0))
        self.assertTrue(np.isclose(tree.min(0, -1), 1.0))
        self.assertTrue(np.isclose(tree.min(2, 4), 3.0))
        self.assertTrue(np.isclose(tree.min(2, 3), 4.0))
        self.assertTrue(np.isclose(tree.min(2, -1), 4.0))
        self.assertTrue(np.isclose(tree.min(3, 4), 3.0))

if __name__ == '__main__':
    unittest.main()
