import collections
import gzip
import logging
import random
import numpy as np
import os
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

def generateRandomAction(max_vel):
    """
        Generate an array of shape (6,) of range [-max_vel, max_vel].
    """
    return np.array([random.random() * max_vel * 2 - max_vel for _ in range(6)])

class SortedDisplayDict(dict):
   def __str__(self):
       return "{" + ", ".join("%r: %r" % (key, self[key]) for key in sorted(self)) + "}"

class ReplayBuffer(object):

    def __init__(self, buffer_size):
        self._buffer_size = buffer_size
        self._count = 0
        self._buffer = deque()
        self._is_bkup = False
        self._stats_estimate_sample_size = 1e4

    def __eq__(self, other):
        if not (self._buffer_size == other._buffer_size and self._count == other._count and self._is_bkup ==\
                other._is_bkup and self._stats_estimate_sample_size == other._stats_estimate_sample_size):
            return False
        for i in range(self._count):
            if np.any(self._buffer[i][0] != other._buffer[i][0]):
                return False
        return True

    def split(self, split_ratio, shuffle=False):
        """
        Spawn two replay buffers from this replay buffer according to the split_ratio
        """
        if shuffle:
            indx = np.random.permutation(self._count)
        else:
            indx = np.arange(self._count)
        replay_buffer_1 = ReplayBuffer(self._buffer_size)
        replay_buffer_2 = ReplayBuffer(self._buffer_size)
        for i in range(self._count):
            if i < int(self._count * split_ratio):
                replay_buffer_1.add(*self._buffer[indx[i]])
            else:
                replay_buffer_2.add(*self._buffer[indx[i]])

        return replay_buffer_1, replay_buffer_2

    def save(self, save_path):
        """
        Use double buffer to provide a safe backup (if pickling is interrupted the save file is permenantly damaged)
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self._is_bkup:
            rb_file_name =  "replay-buffer-0.pklz"
        else:
            rb_file_name =  "replay-buffer-1.pklz"

        rb_save_path = os.path.join(save_path, rb_file_name)

        with gzip.open(rb_save_path, "wb") as f:
            pkl.dump(vars(self), f)

        self._is_bkup = not self._is_bkup
        with open(os.path.join(save_path, "checkpoint"), "w") as f:
            f.write(rb_file_name)

    def iter(self):
        """
        Returns a generator which iterates through deque.
        """
        for exp in self._buffer:
            yield exp

    def mean(self):
        """
        Mean of the state, action and reward data
        An estimate based on a sample of the replay buffer
        """
        current_state_batch, action_batch, reward_batch, _, _= self.sample_batch(self._stats_estimate_sample_size)

        return np.mean(current_state_batch, axis=0), np.mean(action_batch, axis=0), np.mean(reward_batch, axis=0)

    def std(self):
        """
        Standard deviation of the state, action and reward data
        An estimate based on a sample of the replay buffer
        """
        current_state_batch, action_batch, reward_batch, _, _= self.sample_batch(self._stats_estimate_sample_size)

        return np.std(current_state_batch, axis=0), np.std(action_batch, axis=0), np.std(reward_batch, axis=0)

    def load(self, load_path):
        if not os.path.exists(os.path.join(load_path, "checkpoint")):
            raise IOError("No replay buffer checkpoint found")
        with open(os.path.join(load_path, "checkpoint"), "r") as f:
            rb_file_name = f.read()

        with gzip.open(os.path.join(load_path, rb_file_name), "rb") as f:
            states = pkl.load(f)
            self._buffer_size = states["_buffer_size"]
            self._count = states["_count"]
            self._buffer = states["_buffer"]
            self._is_bkup = states["_is_bkup"]

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

    def sample_batch_full(self, batch_size, shuffle=True):
        """
        Sample a minibatch from the buffer.
        This method will iterate through all entries within the buffer
        """
        if shuffle:
            batch_indx = np.random.permutation(self._count)
        else:
            batch_indx = np.arange(self._count)

        num_full_batches = int(self._count / batch_size)
        for batch_num in range(num_full_batches):
            current_state_batch = []
            action_batch = []
            reward_batch = []
            termination_batch = []
            next_state_batch = []

            for i in range(batch_size):
                current_state, action, reward, termination, next_state = self._buffer[batch_indx[batch_num *
                    batch_size + i]]
                current_state_batch.append(current_state)
                action_batch.append(action)
                reward_batch.append(reward)
                termination_batch.append(termination)
                next_state_batch.append(next_state)

            yield np.array(current_state_batch), np.array(action_batch), np.array(reward_batch),\
                        np.array(termination_batch), np.array(next_state_batch)

        # TODO: Yield the remaining data points if any
        # Currently discarding any number of remaining data that is not a multiple of the batch_size
        #current_state_batch = []
        #action_batch = []
        #reward_batch = []
        #termination_batch = []
        #next_state_batch = []

        #for i in range(num_full_batches * batch_size, self._count):
        #    current_state, action, reward, termination, next_state = self._buffer[batch_indx[i]]
        #    current_state_batch.append(current_state)
        #    action_batch.append(action)
        #    reward_batch.append(reward)
        #    termination_batch.append(termination)
        #    next_state_batch.append(next_state)

        #if len(current_state_batch) != 0:
        #    yield np.array(current_state_batch), np.array(action_batch), np.array(reward_batch),\
        #                np.array(termination_batch), np.array(next_state_batch)


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
