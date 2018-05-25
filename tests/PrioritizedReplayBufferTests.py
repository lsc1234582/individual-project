import unittest
import sys
sys.path.insert(0, "../src")
import numpy as np
from Utils import PrioritizedReplayBuffer

class PrioritizedReplayBufferTest(unittest.TestCase):

    def setup(self):
        pass

    def testSamplingFrequencyReflectsPriority(self):
        rb_max_size = 40
        alpha = 0.3
        num_priorities = 10
        num_rep = 2               # Number of repetive entries for each priority
        num_samples = 2000
        minibatch_size = 10

        rb = PrioritizedReplayBuffer(rb_max_size, alpha, debug=True)

        priorities = [0 for _ in range(num_priorities)]
        for i in range(1, num_priorities+1):
            priority = i
            priorities[i-1] = priority
            for _ in range(num_rep):
                rb.add(i, i, i, i, i)

        #sample_freq = [0 for _ in range(num_priorities)]
        for i in range(num_samples):
            os, acs, rs, nos, ds, ws, idx = rb.sample_batch(minibatch_size, 1.0, no_repeat=False)
            for i, o in enumerate(os):
                #sample_freq[o-1] += 1
                rb.update_priorities([idx[i]], [priorities[o-1]])

        sample_freq = rb.debug_get_sample_freq()
        sample_freq_sum = sum(sample_freq)
        sample_freq = list(map((lambda i: i/sample_freq_sum),sample_freq))
        sample_priorities = rb.debug_get_priorities()
        prob_sum = sum(sample_priorities)
        probs = list(map((lambda i : i / prob_sum), sample_priorities))
        #print("hahahaha")
        #print(sample_freq)
        #print(probs)
        #print(priorities)

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

    def testSampleEpisode(self):
        rb_max_size = int(1e3)
        alpha = 0.3
        num_priorities = 10

        rb = PrioritizedReplayBuffer(rb_max_size, alpha)
        for i in range(99):
            rb.add(i, i, i, i, False)
        rb.add(99, 99, 99, 99, True)

        for i in range(100, 249):
            rb.add(i, i, i, i, False)
        rb.add(249, 249, 249, 249, True)

        data_col1 = np.arange(100)
        data_col2 = np.arange(100, 250)
        data_col3 = np.arange(100, 200)

        s0, a0, r0, s1, d, _, _ = rb.sample_episode(1.0)

        self.assertTrue(np.all(s0==data_col1) or np.all(s0==data_col2))
        self.assertTrue(np.all(a0==data_col1) or np.all(s0==data_col2))
        self.assertTrue(np.all(r0==data_col1) or np.all(s0==data_col2))
        self.assertTrue(np.all(s1==data_col1) or np.all(s0==data_col2))

        rb.clear()
        for i in range(99):
            rb.add(i, i, i, i, False)
        rb.add(99, 99, 99, 99, True)

        for i in range(100, 199):
            rb.add(i, i, i, i, False)
        rb.add(199, 199, 199, 199, True)

        s0, a0, r0, s1, d, _, _ = rb.sample_episode(1.0)

        self.assertTrue(np.all(s0==data_col1) or np.all(s0==data_col3))
        self.assertTrue(np.all(a0==data_col1) or np.all(s0==data_col3))
        self.assertTrue(np.all(r0==data_col1) or np.all(s0==data_col3))
        self.assertTrue(np.all(s1==data_col1) or np.all(s0==data_col3))

    def testBatchSizeBiggerThanActualSize(self):
        # NOTE: Should produce assertion error
        rb_max_size = 40
        alpha = 0.3
        num_priorities = 10
        # Number of repetitions for each priority
        num_rep = 1
        num_samples = 5
        minibatch_size = 20

        rb = PrioritizedReplayBuffer(rb_max_size, alpha)

        priorities = [0 for _ in range(num_priorities)]
        for i in range(1, num_priorities+1):
            priority = i
            priorities[i-1] = priority
            for _ in range(num_rep):
                rb.add(i, i, i, i, i)

        #os, acs, rs, nos, ds, ws, idx = rb.sample_batch(minibatch_size, 1.0, no_repeat=False)
        #print("RB SAMPLE")
        #print(os)

    def testNoRepeatedSample(self):
        # Construct replay buffer
        rb_max_size = 40
        alpha = 0.3
        minibatch_size = 10
        num_entries = 10
        # Number of repetitions for the test
        num_rep = 100

        rb = PrioritizedReplayBuffer(rb_max_size, alpha)

        for i in range(num_entries):
            rb.add(i, i, i, i, i)

        # Make the priority of last entry in the buffer really large
        priorities = [1 for _ in range(num_entries)]
        priorities[-1] = 100
        rb.update_priorities(list(range(num_entries)), priorities)

        def hasNoRepeats(arr, elem):
            occurrence = 0
            for e in arr:
                if e == elem:
                    if occurrence > 0:
                        return False
                    occurrence += 1
            return True

        for _ in range(num_rep):
            _, _, _, _, _, _, idx = rb.sample_batch(minibatch_size, 1.0)
            self.assertTrue(hasNoRepeats(idx, num_entries - 1))

    def testNoRepeatSamplingFrequencyReflectsPriority(self):
        """
        No-repeat sampling should still reflect priority although not as much as with-repeat sampling

        """
        rb_max_size = 4000
        alpha = 0.3
        minibatch_size = 100
        num_priorities = 10
        num_samples = 2000

        rb = PrioritizedReplayBuffer(rb_max_size, alpha, debug=True)

        priorities = [0 for _ in range(num_priorities)]
        for i in range(1, num_priorities+1):
            priority = i
            priorities[i-1] = priority
            for _ in range(100):
                rb.add(i, i, i, i, i)

        #sample_freq = [0 for _ in range(num_priorities)]
        for i in range(num_samples):
            os, acs, rs, nos, ds, ws, idx = rb.sample_batch(minibatch_size, 1.0)
            for i, o in enumerate(os):
                #sample_freq[o-1] += 1
                rb.update_priorities([idx[i]], [priorities[o-1]])

        sample_freq = rb.debug_get_sample_freq()
        sample_freq_sum = sum(sample_freq)
        sample_freq = list(map((lambda i: i/sample_freq_sum),sample_freq))
        sample_priorities = rb.debug_get_priorities()
        prob_sum = sum(probs)
        probs = list(map((lambda i : i / prob_sum), sample_priorities))
        #print("hahahaha")
        #print(sample_freq)
        #print(probs)
        #print(priorities)

        self.assertTrue(np.all(np.abs(np.array(probs) - np.array(sample_freq)) < 5e-2))


if __name__ == '__main__':
    unittest.main()

