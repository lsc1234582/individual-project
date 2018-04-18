import collections
import gzip
import logging
import random
import numpy as np
import pickle as pkl
import tensorflow as tf
from collections import deque

def featurize_state(state, scaler):
    """
    Returns the featurized representation for a state.
    """
    scaled = scaler.transform(state)
    return scaled

def getModuleLogger(module_name):
    """
    Universal configuration of logger across all modules.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # Create handlers
    fh = logging.FileHandler("run_info.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    fdh = logging.FileHandler("run_debug.log")
    fdh.setLevel(logging.DEBUG)
    fdh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    logger.addHandler(fh)
    logger.addHandler(fdh)
    logger.addHandler(ch)
    return logger

class SortedDisplayDict(dict):
   def __str__(self):
       return "{" + ", ".join("%r: %r" % (key, self[key]) for key in sorted(self)) + "}"

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self._buffer_size = buffer_size
        self._count = 0
        self._buffer = deque()

    def save(self, save_path):
        with gzip.open(save_path, "wb") as f:
            pkl.dump(vars(self), f)

    def load(self, load_path):
        with gzip.open(load_path, "rb") as f:
            states = pkl.load(f)
            self._buffer_size = states["_buffer_size"]
            self._count = states["_count"]
            self._buffer = states["_buffer"]

    def add(self, current_state, action, reward, termination, next_state):
        experience = [current_state, action, reward, termination, next_state]

        if self._count < self._buffer_size:
            self._buffer.append(experience)
            self._count += 1
        else:
            self._buffer.popleft()
            self._buffer.append(experience)

    def size(self):
        return self._count

    def sample_batch(self, batch_size):
        '''
        batch_size specifies the number of experiences to add
        to the batch. If the replay buffer has less than batch_size
        elements, simply return all of the elements within the buffer.
        Generally, you'll want to wait until the buffer has at least
        batch_size elements before beginning to sample from it.
        '''
        batch = []

        if self._count < batch_size:
            batch = random.sample(self._buffer, self._count)
        else:
            batch = random.sample(self._buffer, batch_size)

        current_state_batch = np.array([_[0] for _ in batch])
        action_batch = np.array([_[1] for _ in batch])
        reward_batch = np.array([_[2] for _ in batch])
        termination_batch = np.array([_[3] for _ in batch])
        next_state_batch = np.array([_[4] for _ in batch])

        return current_state_batch, action_batch, reward_batch, termination_batch, next_state_batch

    def clear(self):
        self._buffer.clear()
        self._count = 0

    def __str__(self):
        return str(vars(self))

class SummaryWriter(object):

    def __init__(self, sess, summary_dir, var_names):
        self._sess = sess
        self._summary_vars = {}
        self._writer = tf.summary.FileWriter(summary_dir, sess.graph)
        with tf.name_scope("summary"):
            for var_name in var_names:
                self._summary_vars[var_name] = tf.Variable(0., trainable=False, name=var_name)
                tf.summary.scalar(var_name, self._summary_vars[var_name])

        self._summary_ops = tf.summary.merge_all()

    def writeSummary(self, var_values, training_step):
        feed_dict = {self._summary_vars[var_name]: var_values[var_name] for var_name in var_values.keys()}
        summary_str = self._sess.run(self._summary_ops, feed_dict=feed_dict)
        self._writer.add_summary(summary_str, training_step)
        self._writer.flush()

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
