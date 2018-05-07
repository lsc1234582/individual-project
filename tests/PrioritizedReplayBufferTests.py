import unittest
import sys
sys.path.insert(0, "../src")
import numpy as np
from Utils import PrioritizedReplayBuffer

class PrioritizedReplayBufferTest(unittest.TestCase):

    def setup(self):
        pass

    def testSamplingFrequencyReflectsPriority(self):
        rb_max_size = 4000
        alpha = 0.3
        minibatch_size = 100
        num_priorities = 10
        num_samples = 2000

        rb = PrioritizedReplayBuffer(rb_max_size, alpha)

        priorities = [0 for _ in range(num_priorities)]
        for i in range(1, num_priorities+1):
            priority = i
            priorities[i-1] = priority
            for _ in range(100):
                rb.add(i, i, i, i, i)

        sample_freq = [0 for _ in range(num_priorities)]
        for i in range(num_samples):
            os, acs, rs, nos, ds, ws, idx = rb.sample_batch(minibatch_size, 1.0)
            sample_priority = []
            for i, o in enumerate(os):
                sample_freq[o-1] += 1
                sample_priority.append(o)
                rb.update_priorities([idx[i]], [priorities[o-1]])

        sample_freq_sum = sum(sample_freq)
        sample_freq = list(map((lambda i: i/sample_freq_sum),sample_freq))
        probs = list(map((lambda i : i ** alpha), priorities))
        prob_sum = sum(probs)
        probs = list(map((lambda i : i / prob_sum), probs))

        self.assertTrue(np.all(np.abs(np.array(probs) - np.array(sample_freq)) < 1e-2))

    def testSaveAndLoad(self):
        rb_max_size = int(1e6)
        alpha = 0.3
        num_priorities = 10

        rb = PrioritizedReplayBuffer(rb_max_size, alpha)

        priorities = [0 for _ in range(num_priorities)]
        for i in range(1, num_priorities+1):
            priority = i
            priorities[i-1] = priority
            for _ in range(100):
                rb.add(i, i, i, i, i)

        rb.save("tmp/PrioritizedRB")

        rb_load = PrioritizedReplayBuffer(rb_max_size, alpha)

        rb_load.load("tmp/PrioritizedRB")

        self.assertEqual(rb, rb_load)

if __name__ == '__main__':
    unittest.main()

