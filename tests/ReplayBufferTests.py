import copy
import unittest
import sys
sys.path.insert(0, "../src")
import numpy as np
from Utils import ReplayBuffer

class ReplayBufferTest(unittest.TestCase):

    def setup(self):
        pass

    def _generateRB(size):
        rb = ReplayBuffer(size)
        for _ in range(size):
            rb.add(np.random.rand(2), np.random.rand(1), np.random.rand(1), np.array(np.random.choice([True, False])), np.random.rand(2))
        return rb

    def testSplitRBPreservesSplitRatio(self):
        original_rb = ReplayBufferTest._generateRB(100)
        rb_1, rb_2 = original_rb.split(0.1)
        self.assertTrue(np.abs(rb_1.size() / original_rb.size() - 0.1) < 1e-3)
        self.assertTrue(np.abs(rb_2.size() / original_rb.size() - 0.9) < 1e-3)

        rb_1, rb_2 = original_rb.split(0)
        self.assertEqual(rb_1.size(), 0)
        self.assertTrue(np.abs(rb_2.size() / original_rb.size() - 1.0) < 1e-3)

        rb_1, rb_2 = original_rb.split(1)
        self.assertTrue(np.abs(rb_1.size() / original_rb.size() - 1.0) < 1e-3)
        self.assertEqual(rb_2.size(), 0)

    def testSplitRBPreservesContent(self):
        original_rb = ReplayBufferTest._generateRB(100)
        original_rb_copy = copy.deepcopy(original_rb)
        rb_1, rb_2 = original_rb.split(0.1)

        self.assertEqual(original_rb, original_rb_copy)
        self.assertTrue(np.abs(rb_2.size() / original_rb.size() - 0.9) < 1e-3)

        reassembled_rb = ReplayBuffer(100)
        for exp in rb_1.iter():
            reassembled_rb.add(*exp)
        for exp in rb_2.iter():
            reassembled_rb.add(*exp)

        self.assertEqual(original_rb, reassembled_rb)

if __name__ == '__main__':
    unittest.main()
