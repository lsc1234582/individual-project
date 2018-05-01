import copy
import unittest
import sys
sys.path.insert(0, "../src")
import numpy as np
from ReplayBufferEnhancer import injectGaussianNoise
from Utils import ReplayBuffer

class ReplayBufferEnhancerTest(unittest.TestCase):

    def setup(self):
        self.original_rb = ReplayBuffer(100)
        self.original_rb.add(np.array([1.0,2.0]), np.array([3.0]), np.array([4.0]), np.array([False]), np.array([2.0,3.0]))
        self.resampled_rb = copy.deepcopy(self.original_rb)

    def _generateRB(size):
        rb = ReplayBuffer(size)
        for _ in range(size):
            rb.add(np.random.rand(2), np.random.rand(1), np.random.rand(1), np.array(np.random.choice([True, False])), np.random.rand(2))
        return rb

    def testInjectGaussianNoiseShouldNotMutateOriginalRB(self):
        original_rb = ReplayBufferEnhancerTest._generateRB(100)
        original_rb_copy = copy.deepcopy(original_rb)
        resampled_rb = copy.deepcopy(original_rb)
        injectGaussianNoise(99, original_rb, resampled_rb, 0.0, 1e-2)

        self.assertEqual(original_rb, original_rb_copy)

    def testInjectGaussianNoiseShouldMutateResampledRB(self):
        original_rb = ReplayBufferEnhancerTest._generateRB(100)
        original_rb_copy = copy.deepcopy(original_rb)
        resampled_rb = copy.deepcopy(original_rb)
        injectGaussianNoise(99, original_rb, resampled_rb, 0.0, 1e-2)

        self.assertNotEqual(original_rb, resampled_rb)

    def testInjectGaussianNoiseShouldPreserveMean(self):
        original_rb = ReplayBufferEnhancerTest._generateRB(100)
        resampled_rb = copy.deepcopy(original_rb)
        injectGaussianNoise(99, original_rb, resampled_rb, 0.0, 1e-2)

        original_state_mean, original_action_mean, original_reward_mean = original_rb.mean()
        resampled_state_mean, resampled_action_mean, resampled_reward_mean = resampled_rb.mean()

        self.assertTrue(np.all(np.abs(original_state_mean - resampled_state_mean) < 1e-3))
        self.assertTrue(np.all(np.abs(original_action_mean - resampled_action_mean) < 1e-3))
        self.assertTrue(np.all(np.abs(original_reward_mean - resampled_reward_mean) < 1e-3))

    def testInjectGaussianNoiseShouldPreserveSTD(self):
        original_rb = ReplayBufferEnhancerTest._generateRB(100)
        resampled_rb = copy.deepcopy(original_rb)
        injectGaussianNoise(99, original_rb, resampled_rb, 0.0, 1e-2)

        original_state_std, original_action_std, original_reward_std = original_rb.std()
        resampled_state_std, resampled_action_std, resampled_reward_std = resampled_rb.std()

        self.assertTrue(np.all(np.abs(original_state_std - resampled_state_std) < 1e-3))
        self.assertTrue(np.all(np.abs(original_action_std - resampled_action_std) < 1e-3))
        self.assertTrue(np.all(np.abs(original_reward_std - resampled_reward_std) < 1e-3))

if __name__ == '__main__':
    unittest.main()
